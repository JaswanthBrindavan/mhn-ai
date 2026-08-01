"""Process one received message end to end.

Claim the item, run the pipeline under a visibility heartbeat, then reach a terminal
state and acknowledge the message. Every branch that ends the work deletes the
message; a transient failure deliberately does NOT, so SQS redelivers it.
"""

import logging
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.integrations.ai.base import AIProvider
from app.integrations.sqs import ReceivedMessage, delete_message
from app.models.ai_results import AiReportClassification
from app.models.enums import RunItemStatus
from app.services import assembly, processing
from app.services.classification import DocumentSection
from app.services.processing import ClaimOutcome
from app.workers.heartbeat import VisibilityHeartbeat
from app.workers.stages import (
    CLASSIFY_STAGE,
    SECTION_PIPELINES,
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
            logger.info("item_rejected", extra={"item_id": str(item_id), "reason": exc.code})
            _ack(sqs, settings, message)
            return Outcome.REJECTED
        except TransientStageError as exc:
            # Leave the item where it is and do NOT delete: redelivery retries it,
            # and the attempt cap in claim_item eventually gives up.
            logger.warning(
                "item_transient_failure",
                extra={"item_id": str(item_id), "reason": str(exc)},
            )
            return Outcome.RETRY

    if outcome is Outcome.COMPLETED:
        _ack(sqs, settings, message)
    elif outcome is Outcome.CANCELLED:
        # cancelled is terminal and set by the API; just drop the message.
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


def _run_pipeline(ctx: StageContext, session: Session) -> Outcome:
    """Classify, then run whatever that section needs, honouring cancellation throughout.

    The pipeline's shape is chosen *after* classification because the section decides it:
    a report is extracted and interpreted, a section document is transcribed and stops.
    """
    if not _run_stage(ctx, session, CLASSIFY_STAGE):
        return Outcome.CANCELLED
    # Re-checked here specifically: routing reads the classification row *unguarded*, and a
    # cancel landing while the stage ran would otherwise be seen as a missing row. The
    # per-stage checks below cover the rest, and the final move/complete is guarded.
    if processing.is_cancelled(session, ctx.item_id):
        return Outcome.CANCELLED

    section = _classified_section(session, ctx)
    pipeline = SECTION_PIPELINES.get(section)
    if pipeline is None:
        # Correctly classified, just not a section this service processes. Routing, not
        # failure: the document stays in unclassified_files with its section recorded.
        raise RejectStageError(
            section.value, f"Document classified as {section.value}, which is not processed"
        )

    for step in pipeline:
        if not _run_stage(ctx, session, step):
            return Outcome.CANCELLED

    if section is DocumentSection.REPORTS:
        # Assemble the content and move it into reports, recording section_row_id and
        # completing the item in one transaction.
        content = assembly.build_content(session, ctx.item_id)
        if processing.move_and_complete(
            session, ctx.item_id, ctx.document_id, content, expected=_IN_PROGRESS
        ):
            logger.info("item_completed", extra={"item_id": str(ctx.item_id)})
            return Outcome.COMPLETED
        # Move guard failed → cancelled between the last stage and here.
        return Outcome.CANCELLED

    # A non-report section: the extraction is stored and the document stays in
    # unclassified_files. Filing it is Spring's call — see docs/document-filing-design.md.
    if processing.complete_item(session, ctx.item_id, expected=_IN_PROGRESS):
        logger.info("item_completed", extra={"item_id": str(ctx.item_id), "section": section.value})
        return Outcome.COMPLETED
    return Outcome.CANCELLED


def _ack(sqs: "SQSClient", settings: Settings, message: ReceivedMessage) -> None:
    delete_message(sqs, settings.sqs_queue_url, message.receipt_handle)


def _terminal_outcome(claim: ClaimOutcome) -> Outcome:
    return {
        ClaimOutcome.NOT_FOUND: Outcome.NOT_FOUND,
        ClaimOutcome.SKIP_TERMINAL: Outcome.SKIPPED_TERMINAL,
        ClaimOutcome.GAVE_UP: Outcome.GAVE_UP,
    }[claim]
