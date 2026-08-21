"""Process one received message end to end.

Claim the item, run the pipeline under a visibility heartbeat, then reach a terminal
state and acknowledge the message. Every branch that ends the work deletes the
message; a transient failure deliberately does NOT, so SQS redelivers it.
"""

import logging
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.integrations.ai.base import AIProvider
from app.integrations.sqs import ReceivedMessage, delete_message
from app.models.ai_results import AiReportClassification
from app.models.enums import RunItemStatus
from app.models.processing import AiProcessingRunItem
from app.services import assembly, filing, identity, processing, section_extraction
from app.services.classification import DocumentSection
from app.services.processing import ClaimOutcome
from app.workers.heartbeat import VisibilityHeartbeat
from app.workers.stages import (
    CLASSIFY_STAGE,
    HANDWRITTEN_PRESCRIPTION_PIPELINE,
    SECTION_PIPELINES,
    PermanentStageError,
    RejectStageError,
    StageContext,
    StageStep,
    TransientStageError,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_sqs.client import SQSClient

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], Session]


class Outcome(StrEnum):
    COMPLETED = "completed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    SKIPPED_TERMINAL = "skipped_terminal"
    NOT_FOUND = "not_found"
    GAVE_UP = "gave_up"
    #: A deterministic failure the attempt cap would only pay to repeat.
    FAILED = "failed"
    #: Left on the queue for redelivery; the only outcome that does not delete.
    RETRY = "retry"


# In-progress states a stage transition may advance from. On a redelivery the item is
# re-claimed to `processing`, so every stage's "expected prior" includes processing.
_IN_PROGRESS = {
    RunItemStatus.PROCESSING.value,
    RunItemStatus.CLASSIFYING.value,
    RunItemStatus.EXTRACTING.value,
    RunItemStatus.GENERATING_INSIGHTS.value,
}


def process_message(
    message: ReceivedMessage,
    *,
    session_factory: SessionFactory,
    s3: "S3Client",
    sqs: "SQSClient",
    ai: AIProvider,
    settings: Settings,
) -> Outcome:
    session = session_factory()
    try:
        return _process(message, session=session, s3=s3, sqs=sqs, ai=ai, settings=settings)
    finally:
        session.close()


def _process(
    message: ReceivedMessage,
    *,
    session: Session,
    s3: "S3Client",
    sqs: "SQSClient",
    ai: AIProvider,
    settings: Settings,
) -> Outcome:
    item_id = message.item_id
    claim = processing.claim_item(session, item_id, max_attempts=settings.max_attempts)

    if claim.outcome is not ClaimOutcome.PROCEED:
        # NOT_FOUND (stale), SKIP_TERMINAL (already done/cancelled), or GAVE_UP
        # (just marked failed) — none should be redelivered.
        if claim.outcome is ClaimOutcome.GAVE_UP:
            # claim_item has just marked the item failed. If the document was already
            # filed, its content still says "classified" and the app would show it as
            # processing for ever.
            filing.mark_content_failed(session, item_id)
        _ack(sqs, settings, message)
        return _terminal_outcome(claim.outcome)

    logger.info(
        "item_claimed",
        extra={
            "item_id": str(item_id),
            "attempt": claim.attempt,
            "receive_count": message.approx_receive_count,
        },
    )

    ctx = StageContext(
        item_id=item_id,
        run_id=message.run_id,
        document_id=message.document_id,
        # "" rather than None so the stages take a plain str: load_source_document turns
        # the empty case into a reject, which is what a missing key means.
        source_key=claim.source_key or "",
        attempt=claim.attempt,
        session=session,
        s3=s3,
        ai=ai,
        settings=settings,
    )

    with VisibilityHeartbeat(
        sqs,
        settings.sqs_queue_url,
        message.receipt_handle,
        settings.sqs_visibility_timeout_seconds,
    ):
        try:
            outcome = _run_pipeline(ctx, session)
        except RejectStageError as exc:
            processing.reject_item(
                session, item_id, code=exc.code, message=exc.message, expected=_IN_PROGRESS
            )
            # No-op when the document was never filed (a section mismatch, or a section
            # with no pipeline); needed when a stage *after* filing rejected.
            filing.mark_content_failed(session, item_id)
            logger.info("item_rejected", extra={"item_id": str(item_id), "reason": exc.code})
            _ack(sqs, settings, message)
            return Outcome.REJECTED
        except PermanentStageError as exc:
            # Deterministic failure — a truncated response fails the same way every
            # time — so spend no further attempts on it. Ends `failed`, not `rejected`:
            # rejection means the document was routed rather than processed, and Spring
            # is told not to show that as an error. This one is one.
            processing.fail_item(
                session, item_id, code=exc.code, message=exc.message, expected=_IN_PROGRESS
            )
            filing.mark_content_failed(session, item_id)
            logger.warning(
                "item_permanent_failure",
                extra={"item_id": str(item_id), "reason": exc.code},
            )
            _ack(sqs, settings, message)
            return Outcome.FAILED
        except TransientStageError as exc:
            # Leave the item where it is and do NOT delete: redelivery retries it,
            # and the attempt cap in claim_item eventually gives up.
            logger.warning(
                "item_transient_failure",
                extra={"item_id": str(item_id), "reason": str(exc)},
            )
            return Outcome.RETRY

    if outcome in (Outcome.COMPLETED, Outcome.CANCELLED, Outcome.REJECTED):
        # All terminal. `cancelled` is set by the API, and `rejected` here is the
        # filed-against-our-classification path, which handles its own bookkeeping rather
        # than raising — see _file_against_classification.
        _ack(sqs, settings, message)
    return outcome


def _run_stage(ctx: StageContext, session: Session, step: StageStep) -> bool:
    """Run one stage. Returns False when the item was cancelled instead."""
    stage_status, stage_fn = step
    if processing.is_cancelled(session, ctx.item_id):
        return False
    if not processing.advance(session, ctx.item_id, to_status=stage_status, expected=_IN_PROGRESS):
        # Guard matched nothing: the item was cancelled or moved. Stop cleanly.
        return False
    stage_fn(ctx)
    return True


def _classified_section(session: Session, ctx: StageContext) -> DocumentSection:
    """The section the classification stage just recorded.

    Transient rather than an unhandled error when the row is absent: ``classify_report``
    persists it before returning, so a gap here means the stage did not really run, and a
    redelivery re-runs it. Letting ``NoResultFound`` escape would bypass both the reject
    and the retry paths.
    """
    value = session.execute(
        select(AiReportClassification.section).where(
            AiReportClassification.run_item_id == ctx.item_id
        )
    ).scalar_one_or_none()
    if value is None:
        raise TransientStageError("no classification recorded for this item")
    return DocumentSection(value)


def _intended_section(session: Session, item_id: UUID) -> DocumentSection | None:
    """The section the user uploaded into, or None for a global upload.

    Never a claim about what the document *is* — only the classifier decides that. It is
    read here solely to catch a disagreement between the two.
    """
    value = session.execute(
        select(AiProcessingRunItem.intended_section).where(AiProcessingRunItem.id == item_id)
    ).scalar_one_or_none()
    return DocumentSection(value) if value else None


def _source_key(session: Session, item_id: UUID) -> str:
    """The document's current S3 key, re-read after filing relocated the object."""
    value = session.execute(
        select(AiProcessingRunItem.source_key).where(AiProcessingRunItem.id == item_id)
    ).scalar_one_or_none()
    return str(value) if value else ""


def _file_against_classification(
    ctx: StageContext, session: Session, *, intended: DocumentSection, detected: DocumentSection
) -> Outcome:
    """The user filed it under one section and we read it as another. File it their way.

    Their choice wins on WHERE, ours wins on WHETHER. The document goes to the section they
    picked — an upload category or a move out of Unclassified is an explicit instruction —
    and the pipeline does not run, because running an insurance policy through the scan
    extractor produces confident nonsense. The disagreement is recorded as a flag, which is
    also what unlocks the "Move to <detected>" action.

    This replaces rejecting-and-leaving-it-in-intake, which read as the safe option and was
    not: `moveUnclassified` publishes nothing and deletes the source object, so the
    re-filing the design told users to do was the one action that made a document
    permanently unprocessable.

    Ends `rejected` with `section_mismatch` as before — it is routing, not an error, and
    Spring is told not to surface it as one. What is new is that `section_row_id` and
    `filed_section` are set, so the caller can see both where it went and where we think it
    belongs.
    """
    if intended not in filing.SECTION_TABLES:
        # `medical_condition`: no table binding here, so we cannot file it and Spring keeps
        # its own mover for it. Unchanged behaviour — stays in intake.
        raise RejectStageError(
            "section_mismatch",
            f"Filed under {intended.value} but classified as {detected.value}",
        )

    section_row_id = filing.file_document(
        session,
        ctx.s3,
        item_id=ctx.item_id,
        document_id=ctx.document_id,
        section=intended,
        content=assembly.build_content(
            session, ctx.item_id, state=assembly.ContentState.CLASSIFIED
        ),
        bucket=ctx.settings.s3_bucket,
        expected=_IN_PROGRESS,
    )
    if section_row_id is None:
        return Outcome.CANCELLED

    section_extraction.record_section_mismatch(ctx, intended, detected)
    filing.write_content(
        session,
        ctx.item_id,
        assembly.build_content(session, ctx.item_id, state=assembly.ContentState.COMPLETE),
    )
    processing.reject_item(
        session,
        ctx.item_id,
        code="section_mismatch",
        message=f"Filed under {intended.value} but classified as {detected.value}",
        expected=_IN_PROGRESS,
    )
    logger.info(
        "item_rejected",
        extra={"item_id": str(ctx.item_id), "reason": "section_mismatch"},
    )
    return Outcome.REJECTED


def _run_pipeline(ctx: StageContext, session: Session) -> Outcome:
    """Classify, file, then run whatever that section needs, honouring cancellation.

    Filing sits *between* classification and the section's stages, not at the end: a user
    who uploaded into a section sees their document there within seconds rather than after
    the 30-80s the AI stages take. Everything after filing updates the filed row's
    ``content``, which is why every terminal path below stamps ``failed`` on it — a filed
    document whose pipeline stops would otherwise read as "still processing" for ever.

    The pipeline's shape is chosen *after* classification because the section decides it:
    a report is extracted and interpreted, a section document is transcribed and stops.
    """
    if not _run_stage(ctx, session, CLASSIFY_STAGE):
        return Outcome.CANCELLED
    # Re-checked here specifically: routing reads the classification row *unguarded*, and a
    # cancel landing while the stage ran would otherwise be seen as a missing row. The
    # per-stage checks below cover the rest, and filing and completion are both guarded.
    if processing.is_cancelled(session, ctx.item_id):
        return Outcome.CANCELLED

    section = _classified_section(session, ctx)

    # Before filing, and before the intended/detected comparison: whether the document is
    # this person's at all is a more fundamental question than which section it belongs
    # in. A document that is not theirs should not be filed anywhere, including into the
    # section they chose.
    identity.gate(ctx, section)

    intended = _intended_section(session, ctx.item_id)
    if intended is not None and intended is not section:
        return _file_against_classification(ctx, session, intended=intended, detected=section)

    pipeline = SECTION_PIPELINES.get(section)
    if section is DocumentSection.PRESCRIPTIONS and ctx.handwriting == "mostly":
        # File it, but read nothing off it. A handwritten page has no text layer, so the
        # name guard cannot reject with it — the one document most likely to be misread is
        # the one where nothing downstream can check the model. The app asks for the
        # pharmacy bill instead, which is printed and lists the same drugs.
        pipeline = HANDWRITTEN_PRESCRIPTION_PIPELINE
    if section is DocumentSection.PRESCRIPTIONS and not ctx.settings.prescriptions_enabled:
        # An emergency stop, no longer a graceful one — the flag defaults ON since
        # 2026-08-18 and Spring lists prescriptions, so the 501 this guarded is gone.
        #
        # What "off" costs now: Spring routes prescriptions through intake, so rejecting
        # here leaves the document in Unclassified rather than in the section the user
        # filed it into. Misfiled, not merely unread. To stop prescriptions gracefully,
        # drop them from Spring's `AiClient.PROCESSABLE` instead — the document then goes
        # straight to its own table, unread, exactly as before this service existed.
        # See the note on `prescriptions_enabled` in config.py.
        pipeline = None
    if pipeline is None:
        # Correctly classified, just not a section this service processes. Routing, not
        # failure: the document stays in unclassified_files with its section recorded.
        raise RejectStageError(
            section.value, f"Document classified as {section.value}, which is not processed"
        )

    section_row_id = filing.file_document(
        session,
        ctx.s3,
        item_id=ctx.item_id,
        document_id=ctx.document_id,
        section=section,
        content=assembly.build_content(
            session, ctx.item_id, state=assembly.ContentState.CLASSIFIED
        ),
        bucket=ctx.settings.s3_bucket,
        expected=_IN_PROGRESS,
    )
    if section_row_id is None:
        # The guard matched nothing (cancelled or moved underneath us) and nothing was
        # filed, so there is no content to stamp.
        return Outcome.CANCELLED
    # Filing relocated the object; the in-memory key is now stale and every stage below
    # loads the document through it.
    ctx.source_key = _source_key(session, ctx.item_id)

    for step in pipeline:
        if not _run_stage(ctx, session, step):
            filing.mark_content_failed(session, ctx.item_id)
            return Outcome.CANCELLED

    filing.write_content(
        session,
        ctx.item_id,
        assembly.build_content(session, ctx.item_id, state=assembly.ContentState.COMPLETE),
        extra=filing.extra_columns(session, ctx.item_id, section),
    )
    if processing.complete_item(session, ctx.item_id, expected=_IN_PROGRESS):
        logger.info("item_completed", extra={"item_id": str(ctx.item_id), "section": section.value})
        return Outcome.COMPLETED
    filing.mark_content_failed(session, ctx.item_id)
    return Outcome.CANCELLED


def _ack(sqs: "SQSClient", settings: Settings, message: ReceivedMessage) -> None:
    delete_message(sqs, settings.sqs_queue_url, message.receipt_handle)


def _terminal_outcome(claim: ClaimOutcome) -> Outcome:
    return {
        ClaimOutcome.NOT_FOUND: Outcome.NOT_FOUND,
        ClaimOutcome.SKIP_TERMINAL: Outcome.SKIPPED_TERMINAL,
        ClaimOutcome.GAVE_UP: Outcome.GAVE_UP,
    }[claim]
