"""Document-processing run business logic.

Two invariants worth stating up front.

**Idempotency is enforced by the database.** A partial unique index permits at most one
in-flight item per document, so two concurrent submissions cannot both create work. One
raises ``IntegrityError`` and we reuse the winner's row. A read-then-write check alone
loses that race.

**Messages are published only after the transaction commits.** SQS delivery can be
faster than a transaction; publishing first lets a worker receive an item id that no
committed row matches yet. Publishing after means the worst case is a committed item
that never got a message — visible as `pending`, and recoverable by the stale-item
sweep. That failure is recoverable; the other is a phantom.
"""

import logging
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ApiError
from app.integrations.s3 import ObjectMetadata
from app.integrations.sqs import PublishError, publish_processing_item
from app.models.ai_results import AiReportClassification
from app.models.enums import ACTIVE_STATUSES, CANCELLABLE_STATUSES, RunItemStatus
from app.models.processing import (
    ACTIVE_STATUS_PREDICATE,
    AiProcessingRun,
    AiProcessingRunItem,
)
from app.models.spring import unclassified_files
from app.schemas.results import DocumentType
from app.schemas.runs import (
    CancelRunResponse,
    CreateRunRequest,
    CreateRunResponse,
    RunItemResponse,
    RunProgress,
    RunResponse,
    SubmitOutcome,
    SubmittedItem,
)
from app.services import filing
from app.services.classification import DOCUMENT_TYPE_BY_SECTION
from app.services.source_validation import (
    SourceObjectUnavailableError,
    ValidationFailure,
    validate_source_object,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_sqs.client import SQSClient

logger = logging.getLogger(__name__)

_ACTIVE = {status.value for status in ACTIVE_STATUSES}
_CANCELLABLE = {status.value for status in CANCELLABLE_STATUSES}

#: How many source files to HeadObject at once. Small enough not to hammer S3 or
#: exhaust the botocore connection pool, large enough that a big batch is not serial.
_VALIDATION_CONCURRENCY = 8


def _document_filepaths(session: Session, document_ids: list[int]) -> dict[int, str]:
    """Existence + source key lookup against unclassified_files. A sanity check."""
    rows = session.execute(
        select(unclassified_files.c.id, unclassified_files.c.filepath).where(
            unclassified_files.c.id.in_(document_ids)
        )
    ).all()
    return {int(row.id): row.filepath for row in rows}


def _active_items(session: Session, document_ids: list[int]) -> dict[int, AiProcessingRunItem]:
    """In-flight item per document, for the whole batch in one query."""
    rows = session.execute(
        select(AiProcessingRunItem).where(
            AiProcessingRunItem.document_id.in_(document_ids),
            AiProcessingRunItem.status.in_(_ACTIVE),
        )
    ).scalars()
    return {item.document_id: item for item in rows}


def _latest_items(session: Session, document_ids: list[int]) -> dict[int, AiProcessingRunItem]:
    """Most recent item per document, for the whole batch in one query.

    ``DISTINCT ON`` is PostgreSQL-specific, which is fine — this service targets
    PostgreSQL — and avoids one query per document on a 500-document submission.
    """
    rows = session.execute(
        select(AiProcessingRunItem)
        .where(AiProcessingRunItem.document_id.in_(document_ids))
        .distinct(AiProcessingRunItem.document_id)
        .order_by(AiProcessingRunItem.document_id, AiProcessingRunItem.created_at.desc())
    ).scalars()
    return {item.document_id: item for item in rows}


def _validate_sources(
    s3: "S3Client", settings: Settings, targets: dict[int, str]
) -> dict[int, tuple[ObjectMetadata | None, ValidationFailure | None]]:
    """HeadObject every candidate, a few at a time.

    Sequentially this is one network round trip per document — on a 500-document batch
    that alone is minutes of wall clock in an endpoint expected to answer quickly.
    boto3 clients are thread-safe for API calls, so a small pool is enough.
    """
    if not targets:
        return {}

    if len(targets) == 1:
        document_id, filepath = next(iter(targets.items()))
        return {document_id: validate_source_object(s3, settings, filepath)}

    workers = min(_VALIDATION_CONCURRENCY, len(targets))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="validate") as pool:
        futures = {
            pool.submit(validate_source_object, s3, settings, filepath): document_id
            for document_id, filepath in targets.items()
        }
        # A transient S3 failure in any worker propagates, so the caller answers 503
        # rather than permanently rejecting documents over a blip.
        return {futures[future]: future.result() for future in as_completed(futures)}


@dataclass
class _Resolved:
    """What a submitted document ended up mapped to."""

    item_id: uuid.UUID
    document_id: int
    run_id: uuid.UUID
    status: str
    error_code: str | None
    outcome: SubmitOutcome


@dataclass
class _Plan:
    """Decisions made before touching the database."""

    #: Documents answered from existing rows: in flight, or completed without force.
    reused: dict[int, AiProcessingRunItem]
    outcomes: dict[int, SubmitOutcome]
    #: Documents needing a new item, as insertable column dicts.
    new_rows: dict[int, dict[str, Any]]


def _plan_items(
    document_ids: list[int],
    active: dict[int, AiProcessingRunItem],
    latest: dict[int, AiProcessingRunItem],
    validated: dict[int, tuple[ObjectMetadata | None, ValidationFailure | None]],
    force_reprocess: bool,
    intended: dict[int, str | None],
    source_keys: dict[int, str],
) -> _Plan:
    """Pure decision step: no I/O, so the rules are easy to test and to read."""
    reused: dict[int, AiProcessingRunItem] = {}
    outcomes: dict[int, SubmitOutcome] = {}
    new_rows: dict[int, dict[str, Any]] = {}

    for document_id in document_ids:
        in_flight = active.get(document_id)
        if in_flight is not None:
            # Already running, and already has a queue message. force_reprocess
            # concerns finished results, not work still under way.
            reused[document_id] = in_flight
            outcomes[document_id] = SubmitOutcome.REUSED
            continue

        previous = latest.get(document_id)
        if (
            previous is not None
            and previous.status == RunItemStatus.COMPLETED.value
            and not force_reprocess
        ):
            # Never overwrite a completed result without an explicit force_reprocess.
            reused[document_id] = previous
            outcomes[document_id] = SubmitOutcome.ALREADY_COMPLETED
            continue

        meta, failure = validated.get(document_id, (None, None))
        # Every row carries the same keys: a multi-row VALUES clause cannot mix
        # differing column sets, and omitting one here fails at compile time.
        new_rows[document_id] = {
            "document_id": document_id,
            "status": (
                RunItemStatus.REJECTED.value
                if failure is not None
                # Terminal with a reason, rather than failing the whole batch.
                else RunItemStatus.PENDING.value
            ),
            "content_hash": meta.etag if meta is not None else None,
            "last_error_code": failure.code if failure is not None else None,
            "last_error_message": failure.message if failure is not None else None,
            "intended_section": intended.get(document_id),
            # Set here because the key is already loaded for validation. From now on the
            # pipeline reads the document through this, not through unclassified_files.
            "source_key": source_keys.get(document_id),
        }

    return _Plan(reused=reused, outcomes=outcomes, new_rows=new_rows)


def _insert_new_items(
    session: Session, run_id: uuid.UUID, new_rows: dict[int, dict[str, Any]]
) -> dict[int, Any]:
    """Insert every new item in one statement, tolerating the idempotency race.

    ``ON CONFLICT DO NOTHING`` replaces a per-item SAVEPOINT/flush. That matters twice
    over: it is one round trip instead of three per document on a batch of up to 500,
    and it removes the failure mode where catching IntegrityError and calling
    ``session.rollback()`` would discard the whole transaction — run row and all
    previously created items — while the response still reported their ids.
    """
    if not new_rows:
        return {}

    statement = (
        pg_insert(AiProcessingRunItem)
        .values([{**row, "run_id": run_id} for row in new_rows.values()])
        # The predicate must match the partial index exactly to target it.
        .on_conflict_do_nothing(
            index_elements=[AiProcessingRunItem.document_id],
            index_where=text(ACTIVE_STATUS_PREDICATE),
        )
        .returning(
            AiProcessingRunItem.id,
            AiProcessingRunItem.document_id,
            AiProcessingRunItem.status,
            AiProcessingRunItem.last_error_code,
        )
    )
    return {row.document_id: row for row in session.execute(statement).all()}


def _mark_queued(session: Session, item_ids: set[uuid.UUID]) -> None:
    session.execute(
        update(AiProcessingRunItem)
        .where(
            AiProcessingRunItem.id.in_(item_ids),
            # Only advance from pending: a worker may already have picked the item
            # up and moved it on before this update lands.
            AiProcessingRunItem.status == RunItemStatus.PENDING.value,
        )
        .values(status=RunItemStatus.QUEUED.value)
    )
    session.commit()


def create_run(
    session: Session,
    payload: CreateRunRequest,
    request_id: str | None,
    *,
    s3: "S3Client",
    sqs: "SQSClient",
    settings: Settings,
) -> CreateRunResponse:
    # Deduplicate while preserving caller order. First occurrence wins for the intended
    # section: a repeated id in one request is one document, not two.
    intended: dict[int, str | None] = {}
    for document in payload.documents:
        intended.setdefault(
            document.document_id,
            document.intended_section.value if document.intended_section else None,
        )
    unique_ids = list(intended)

    filepaths = _document_filepaths(session, unique_ids)
    missing = [document_id for document_id in unique_ids if document_id not in filepaths]
    if missing:
        raise ApiError(
            404,
            "document_not_found",
            "One or more documents do not exist",
            {"missing_document_ids": missing},
        )

    run = AiProcessingRun(
        requested_by_user_id=payload.requested_by_user_id,
        caller="spring",
        request_id=request_id,
        force_reprocess=payload.force_reprocess,
    )
    session.add(run)
    session.flush()  # assign run.id without committing
    # Captured now: commit() may expire the instance, and re-reading these later
    # would cost an extra round trip for no reason.
    run_id, run_created_at = run.id, run.created_at

    # Two queries for the whole batch rather than two per document.
    active = _active_items(session, unique_ids)
    latest = _latest_items(session, unique_ids)

    # Only documents that will actually produce new work need their source validated;
    # reused and already-completed ones are answered from the database alone.
    needs_validation = {
        document_id: filepaths[document_id]
        for document_id in unique_ids
        if document_id not in active
        and not (
            (existing := latest.get(document_id)) is not None
            and existing.status == RunItemStatus.COMPLETED.value
            and not payload.force_reprocess
        )
    }
    try:
        validated = _validate_sources(s3, settings, needs_validation)
    except SourceObjectUnavailableError as exc:
        raise ApiError(
            503,
            "source_storage_unavailable",
            "Could not verify source files; retry shortly",
        ) from exc

    plan = _plan_items(
        unique_ids, active, latest, validated, payload.force_reprocess, intended, filepaths
    )
    created = _insert_new_items(session, run_id, plan.new_rows)

    # Rows the insert did not return lost the idempotency race to a concurrent
    # submission. ON CONFLICT DO NOTHING means no exception and no lost transaction:
    # look up the winners and reuse them.
    losers = [document_id for document_id in plan.new_rows if document_id not in created]
    winners = _active_items(session, losers) if losers else {}

    resolved: dict[int, _Resolved] = {}
    publishable: list[_Resolved] = []
    for document_id in unique_ids:
        if document_id in plan.reused:
            existing = plan.reused[document_id]
            resolved[document_id] = _Resolved(
                item_id=existing.id,
                document_id=document_id,
                run_id=existing.run_id,
                status=existing.status,
                error_code=existing.last_error_code,
                outcome=plan.outcomes[document_id],
            )
        elif document_id in created:
            row = created[document_id]
            entry = _Resolved(
                item_id=row.id,
                document_id=document_id,
                run_id=run_id,
                status=row.status,
                error_code=row.last_error_code,
                outcome=SubmitOutcome.CREATED,
            )
            resolved[document_id] = entry
            if row.status == RunItemStatus.PENDING.value:
                publishable.append(entry)
        else:
            winner = winners.get(document_id)
            if winner is None:  # pragma: no cover - the winner turned terminal instantly
                raise ApiError(409, "submission_conflict", "Document is already being processed")
            resolved[document_id] = _Resolved(
                item_id=winner.id,
                document_id=document_id,
                run_id=winner.run_id,
                status=winner.status,
                error_code=winner.last_error_code,
                outcome=SubmitOutcome.REUSED,
            )

    # Commit before publishing: a worker must never see an item id that is not
    # yet committed.
    session.commit()

    published = _publish(sqs, settings, publishable)
    if published:
        _mark_queued(session, published)
        for entry in publishable:
            if entry.item_id in published:
                entry.status = RunItemStatus.QUEUED.value

    return CreateRunResponse(
        run_id=run_id,
        created_at=run_created_at,
        items=[
            SubmittedItem(
                document_id=document_id,
                item_id=resolved[document_id].item_id,
                status=resolved[document_id].status,
                outcome=resolved[document_id].outcome,
                error_code=resolved[document_id].error_code,
            )
            for document_id in unique_ids
        ],
    )


def _publish(
    sqs: "SQSClient",
    settings: Settings,
    entries: list["_Resolved"],
) -> set[uuid.UUID]:
    """Enqueue committed items. Returns the ids that made it onto the queue.

    A publish failure is not fatal. The item stays `pending`, which the stale-item
    sweep treats as retryable — better than failing a request whose work is already
    durably recorded.
    """
    if not entries:
        return set()

    if not settings.sqs_queue_url:
        logger.error("publish_skipped_no_queue_configured", extra={"item_count": len(entries)})
        return set()

    published: set[uuid.UUID] = set()
    for entry in entries:
        try:
            message_id = publish_processing_item(
                sqs,
                settings.sqs_queue_url,
                item_id=entry.item_id,
                run_id=entry.run_id,
                document_id=entry.document_id,
                attempt=0,
            )
        except PublishError as exc:
            # Identifiers only -- never the document or the message body.
            logger.error(
                "publish_failed", extra={"item_id": str(entry.item_id), "reason": str(exc)}
            )
            continue
        logger.info(
            "item_published", extra={"item_id": str(entry.item_id), "message_id": message_id}
        )
        published.add(entry.item_id)

    return published


def _document_types(session: Session, item_ids: list[uuid.UUID]) -> dict[uuid.UUID, DocumentType]:
    """The URL type each classified item is readable under, in one query.

    This is what makes the typed result routes usable: without it a caller would know a
    document is finished but not which ``/v1/documents/{type}/...`` URL to call. Sections
    with no addressable type are simply absent from the mapping.
    """
    if not item_ids:
        return {}
    rows = session.execute(
        select(AiReportClassification.run_item_id, AiReportClassification.section).where(
            AiReportClassification.run_item_id.in_(item_ids)
        )
    ).all()
    return {
        item_id: DOCUMENT_TYPE_BY_SECTION[section]
        for item_id, section in rows
        if section in DOCUMENT_TYPE_BY_SECTION
    }


def get_run(session: Session, run_id: uuid.UUID) -> RunResponse:
    run = session.get(AiProcessingRun, run_id)
    if run is None:
        raise ApiError(404, "run_not_found", "Processing run not found")

    counts = Counter(item.status for item in run.items)
    progress = RunProgress(total=len(run.items), **dict(counts))
    finished = not any(item.status in _ACTIVE for item in run.items)
    types = _document_types(session, [item.id for item in run.items])

    return RunResponse(
        run_id=run_id,
        caller=run.caller,
        requested_by_user_id=run.requested_by_user_id,
        force_reprocess=run.force_reprocess,
        created_at=run.created_at,
        updated_at=run.updated_at,
        finished=finished,
        progress=progress,
        items=[
            RunItemResponse.model_validate(item).model_copy(
                update={"document_type": types.get(item.id)}
            )
            for item in run.items
        ],
    )


def cancel_run(session: Session, run_id: uuid.UUID) -> CancelRunResponse:
    """Cancel every in-flight item. Terminal items are left untouched.

    Messages already on the queue cannot be recalled, so workers must re-check item
    status before each stage and stop when they see `cancelled`.
    """
    run = session.get(AiProcessingRun, run_id)
    if run is None:
        raise ApiError(404, "run_not_found", "Processing run not found")

    cancellable = [item.id for item in run.items if item.status in _CANCELLABLE]
    unaffected = [item.id for item in run.items if item.status not in _CANCELLABLE]

    if cancellable:
        cancelled = (
            session.execute(
                update(AiProcessingRunItem)
                .where(
                    AiProcessingRunItem.id.in_(cancellable),
                    # Re-check inside the UPDATE: an item may have advanced to a terminal
                    # state between the read above and this write.
                    AiProcessingRunItem.status.in_(_CANCELLABLE),
                )
                .values(status=RunItemStatus.CANCELLED.value)
                .returning(AiProcessingRunItem.id)
            )
            .scalars()
            .all()
        )
        session.commit()
        # A document filed mid-pipeline keeps `content.ai.state == "classified"` until
        # something says otherwise, and the app reads that as "still processing". No worker
        # will do it here: the one holding this item may be between deliveries (a transient
        # failure left the message on the queue), and when it is redelivered `claim_item`
        # skips a cancelled item without entering the pipeline at all.
        #
        # RETURNING, not `cancellable`: an item that raced to `completed` between the read
        # and the UPDATE is not in this set, so its finished content is never overwritten.
        for item_id in cancelled:
            filing.mark_content_failed(session, item_id)

    return CancelRunResponse(
        run_id=run_id,
        cancelled_item_ids=cancellable,
        unaffected_item_ids=unaffected,
    )
