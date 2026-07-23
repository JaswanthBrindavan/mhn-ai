"""Process one received message end to end.

Claim the item, run the pipeline under a visibility heartbeat, then reach a terminal
state and acknowledge the message. Every branch that ends the work deletes the
message; a transient failure deliberately does NOT, so SQS redelivers it.
"""

import logging
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.integrations.sqs import ReceivedMessage, delete_message
from app.models.enums import RunItemStatus
from app.services import processing
from app.services.processing import ClaimOutcome
from app.workers.heartbeat import VisibilityHeartbeat
from app.workers.stages import STAGE_SEQUENCE, RejectStageError, StageContext, TransientStageError

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
    settings: Settings,
) -> Outcome:
    session = session_factory()
    try:
        return _process(message, session=session, s3=s3, sqs=sqs, settings=settings)
    finally:
        session.close()


def _process(
    message: ReceivedMessage,
    *,
    session: Session,
    s3: "S3Client",
    sqs: "SQSClient",
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
        report_id=message.report_id,
        session=session,
        s3=s3,
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


def _run_pipeline(ctx: StageContext, session: Session) -> Outcome:
    """Advance through every stage, honouring cancellation between and around them."""
    for stage_status, stage_fn in STAGE_SEQUENCE:
        if processing.is_cancelled(session, ctx.item_id):
            return Outcome.CANCELLED

        if not processing.advance(
            session, ctx.item_id, to_status=stage_status, expected=_IN_PROGRESS
        ):
            # Guard matched nothing: the item was cancelled or moved. Stop cleanly.
            return Outcome.CANCELLED

        stage_fn(ctx)

    if processing.complete_item(session, ctx.item_id, expected=_IN_PROGRESS):
        logger.info("item_completed", extra={"item_id": str(ctx.item_id)})
        return Outcome.COMPLETED
    # Completion guard failed → cancelled between the last stage and here.
    return Outcome.CANCELLED


def _ack(sqs: "SQSClient", settings: Settings, message: ReceivedMessage) -> None:
    delete_message(sqs, settings.sqs_queue_url, message.receipt_handle)


def _terminal_outcome(claim: ClaimOutcome) -> Outcome:
    return {
        ClaimOutcome.NOT_FOUND: Outcome.NOT_FOUND,
        ClaimOutcome.SKIP_TERMINAL: Outcome.SKIPPED_TERMINAL,
        ClaimOutcome.GAVE_UP: Outcome.GAVE_UP,
    }[claim]
