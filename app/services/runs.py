"""Report-processing run business logic.

Two invariants worth stating up front.

**Idempotency is enforced by the database.** A partial unique index permits at most one
in-flight item per report, so two concurrent submissions cannot both create work. One
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
from typing import TYPE_CHECKING

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ApiError
from app.integrations.sqs import PublishError, publish_processing_item
from app.models.enums import ACTIVE_STATUSES, CANCELLABLE_STATUSES, RunItemStatus
from app.models.processing import AiProcessingRun, AiProcessingRunItem
from app.models.spring import reports
from app.schemas.runs import (
    CancelRunResponse,
    CreateRunRequest,
    CreateRunResponse,
    RunProgress,
    RunResponse,
    SubmitOutcome,
    SubmittedItem,
)
from app.services.source_validation import SourceObjectUnavailableError, validate_source_object

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_sqs.client import SQSClient

logger = logging.getLogger(__name__)

_ACTIVE = {status.value for status in ACTIVE_STATUSES}
_CANCELLABLE = {status.value for status in CANCELLABLE_STATUSES}


def _report_filepaths(session: Session, report_ids: list[int]) -> dict[int, str]:
    """Existence + source key lookup. A sanity check, NOT an access-control check."""
    rows = session.execute(
        select(reports.c.id, reports.c.filepath).where(reports.c.id.in_(report_ids))
    ).all()
    return {int(row.id): row.filepath for row in rows}


def _latest_item_for_report(session: Session, report_id: int) -> AiProcessingRunItem | None:
    return session.execute(
        select(AiProcessingRunItem)
        .where(AiProcessingRunItem.report_id == report_id)
        .order_by(AiProcessingRunItem.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def _active_item_for_report(session: Session, report_id: int) -> AiProcessingRunItem | None:
    return session.execute(
        select(AiProcessingRunItem).where(
            AiProcessingRunItem.report_id == report_id,
            AiProcessingRunItem.status.in_(_ACTIVE),
        )
    ).scalar_one_or_none()


def create_run(
    session: Session,
    payload: CreateRunRequest,
    request_id: str | None,
    *,
    s3: "S3Client",
    sqs: "SQSClient",
    settings: Settings,
) -> CreateRunResponse:
    # Deduplicate while preserving caller order, so a repeated id in one request
    # cannot try to create two items and trip the unique index against itself.
    unique_ids = list(dict.fromkeys(payload.report_ids))

    filepaths = _report_filepaths(session, unique_ids)
    missing = [report_id for report_id in unique_ids if report_id not in filepaths]
    if missing:
        raise ApiError(
            404,
            "report_not_found",
            "One or more reports do not exist",
            {"missing_report_ids": missing},
        )

    run = AiProcessingRun(
        requested_by_user_id=payload.requested_by_user_id,
        caller="spring",
        request_id=request_id,
        force_reprocess=payload.force_reprocess,
    )
    session.add(run)
    session.flush()  # assign run.id without committing

    outcomes: dict[int, SubmitOutcome] = {}
    items: dict[int, AiProcessingRunItem] = {}
    publishable: list[AiProcessingRunItem] = []

    for report_id in unique_ids:
        item, outcome = _submit_one(
            session,
            run,
            report_id,
            filepaths[report_id],
            force_reprocess=payload.force_reprocess,
            s3=s3,
            settings=settings,
        )
        items[report_id] = item
        outcomes[report_id] = outcome
        if outcome is SubmitOutcome.CREATED and item.status == RunItemStatus.PENDING.value:
            publishable.append(item)

    # Commit before publishing: a worker must never see an item id that is not
    # yet committed.
    session.commit()

    _publish(session, sqs, settings, publishable)

    session.refresh(run)
    return CreateRunResponse(
        run_id=run.id,
        created_at=run.created_at,
        items=[
            SubmittedItem(
                report_id=report_id,
                item_id=items[report_id].id,
                status=items[report_id].status,
                outcome=outcomes[report_id],
                error_code=items[report_id].last_error_code,
            )
            for report_id in unique_ids
        ],
    )


def _submit_one(
    session: Session,
    run: AiProcessingRun,
    report_id: int,
    filepath: str,
    *,
    force_reprocess: bool,
    s3: "S3Client",
    settings: Settings,
) -> tuple[AiProcessingRunItem, SubmitOutcome]:
    """Create, or safely reuse, the item for one report."""
    active = _active_item_for_report(session, report_id)
    if active is not None:
        # Already in flight, and already has a queue message. Reuse rather than
        # process the same report twice; force_reprocess concerns finished results.
        return active, SubmitOutcome.REUSED

    latest = _latest_item_for_report(session, report_id)
    if (
        latest is not None
        and latest.status == RunItemStatus.COMPLETED.value
        and not force_reprocess
    ):
        # Never overwrite a completed result without an explicit force_reprocess.
        return latest, SubmitOutcome.ALREADY_COMPLETED

    # Validate the source file before spending any AI budget on it. A transient S3
    # failure propagates rather than permanently rejecting a valid report.
    try:
        meta, failure = validate_source_object(s3, settings, filepath)
    except SourceObjectUnavailableError as exc:
        raise ApiError(
            503,
            "source_storage_unavailable",
            "Could not verify source files; retry shortly",
        ) from exc

    item = AiProcessingRunItem(run_id=run.id, report_id=report_id)
    if failure is not None:
        # Terminal: recorded with a reason instead of failing the whole batch.
        item.status = RunItemStatus.REJECTED.value
        item.last_error_code = failure.code
        item.last_error_message = failure.message
    else:
        item.status = RunItemStatus.PENDING.value
        item.content_hash = meta.etag if meta else None

    session.add(item)
    try:
        session.flush()
    except IntegrityError:
        # Lost a race with a concurrent submission. The unique index did its job;
        # roll back and reuse whatever the winner created.
        session.rollback()
        session.add(run)
        winner = _active_item_for_report(session, report_id)
        if winner is None:  # pragma: no cover - only on an unrelated constraint
            raise
        return winner, SubmitOutcome.REUSED

    return item, SubmitOutcome.CREATED


def _publish(
    session: Session,
    sqs: "SQSClient",
    settings: Settings,
    items: list[AiProcessingRunItem],
) -> None:
    """Enqueue committed items and advance them to `queued`.

    A publish failure is not fatal. The item stays `pending`, which the stale-item
    sweep treats as retryable — better than failing a request whose work is already
    durably recorded.
    """
    if not items:
        return

    if not settings.sqs_queue_url:
        logger.error(
            "publish_skipped_no_queue_configured",
            extra={"item_count": len(items)},
        )
        return

    published: list[uuid.UUID] = []
    for item in items:
        try:
            message_id = publish_processing_item(
                sqs,
                settings.sqs_queue_url,
                item_id=item.id,
                run_id=item.run_id,
                report_id=item.report_id,
                attempt=item.attempt_count,
            )
        except PublishError as exc:
            # Identifiers only -- never the report or the message body.
            logger.error(
                "publish_failed",
                extra={"item_id": str(item.id), "reason": str(exc)},
            )
            continue
        logger.info("item_published", extra={"item_id": str(item.id), "message_id": message_id})
        published.append(item.id)

    if published:
        session.execute(
            update(AiProcessingRunItem)
            .where(
                AiProcessingRunItem.id.in_(published),
                # Only advance from pending: a worker may already have picked the
                # item up and moved it on before this update lands.
                AiProcessingRunItem.status == RunItemStatus.PENDING.value,
            )
            .values(status=RunItemStatus.QUEUED.value)
        )
        session.commit()


def get_run(session: Session, run_id: uuid.UUID) -> RunResponse:
    run = session.get(AiProcessingRun, run_id)
    if run is None:
        raise ApiError(404, "run_not_found", "Processing run not found")

    counts = Counter(item.status for item in run.items)
    progress = RunProgress(total=len(run.items), **dict(counts))
    finished = not any(item.status in _ACTIVE for item in run.items)

    return RunResponse(
        run_id=run.id,
        caller=run.caller,
        requested_by_user_id=run.requested_by_user_id,
        force_reprocess=run.force_reprocess,
        created_at=run.created_at,
        updated_at=run.updated_at,
        finished=finished,
        progress=progress,
        items=run.items,  # coerced by RunItemResponse's from_attributes config
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
        session.execute(
            update(AiProcessingRunItem)
            .where(
                AiProcessingRunItem.id.in_(cancellable),
                # Re-check inside the UPDATE: an item may have advanced to a terminal
                # state between the read above and this write.
                AiProcessingRunItem.status.in_(_CANCELLABLE),
            )
            .values(status=RunItemStatus.CANCELLED.value)
        )
        session.commit()

    return CancelRunResponse(
        run_id=run_id,
        cancelled_item_ids=cancellable,
        unaffected_item_ids=unaffected,
    )
