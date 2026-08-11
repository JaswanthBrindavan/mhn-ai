"""Re-queue work that stopped without finishing.

Three mechanisms already recover the ordinary cases: the visibility heartbeat keeps a
long stage's message owned, SQS redelivers after the visibility timeout, and the attempt
cap in ``claim_item`` eventually gives up. This is the sweep for what none of them see —
an item that is not terminal and has **no message in flight**, so nothing will ever come
back for it.

Two ways that happens, and the second is the common one here:

* The API committed the item and then died before publishing (it stays ``pending``), or
  the publish itself failed — that one is now handled at source, ending ``failed``.
* **A message was published and another environment's worker ate it.** Local and
  production share one queue: whichever consumer wins the receive looks the item up in
  *its own* database, does not find it, and deletes the message. Our item stays
  ``queued`` for ever, and because it was never classified the document is never filed —
  so it sits in ``unclassified_files`` and the app shows it under the section the user
  chose, pending, with nothing ever generated.

**The reaper survives that; it does not fix it.** One queue per environment removes the
cause. Re-publishing puts the message back into the same race, which is why the bound
below matters more here than it looks.

**Bounding it.** An item stuck at ``queued`` has ``attempt_count = 0`` and would keep it:
the counter only moves in ``claim_item``, which never runs for a message that was eaten.
So ``MAX_ATTEMPTS`` does not bound this on its own, and a naive sweep re-publishes the
same document every interval for ever. The reaper therefore **increments
``attempt_count`` itself** and gives up at the cap exactly as ``claim_item`` does. That
deliberately reuses the existing counter rather than adding a column: Flyway owns this
schema now, so a new column is a migration in someone else's repo plus local/test drift,
to hold a number the existing one can carry. The cost is that a document which is both
re-queued and flaky spends its attempts slightly faster — which fails toward a terminal,
visible state rather than toward a loop.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.integrations.sqs import PublishError, publish_processing_item
from app.models.enums import ACTIVE_STATUSES, RunItemStatus
from app.models.processing import AiProcessingRunItem
from app.services import filing

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_sqs.client import SQSClient

logger = logging.getLogger(__name__)

#: Most items to touch in one sweep. A cap rather than the whole backlog so a sweep can
#: never hold locks over an unbounded set, and so a pathological state cannot produce a
#: thousand SQS publishes in one tick. The next sweep takes the next batch.
_BATCH = 50

_ACTIVE = {status.value for status in ACTIVE_STATUSES}


def sweep_stale_items(
    session: Session,
    sqs: "SQSClient",
    settings: Settings,
) -> int:
    """Re-queue (or give up on) items that have stopped moving. Returns how many.

    Safe to run concurrently in every worker replica: the candidate select takes
    ``FOR UPDATE SKIP LOCKED``, so two sweepers never pick the same row, and a row an
    active worker is claiming is locked by ``claim_item``'s own ``FOR UPDATE``.
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=settings.stale_item_timeout_seconds)

    candidates = session.execute(
        select(
            AiProcessingRunItem.id,
            AiProcessingRunItem.run_id,
            AiProcessingRunItem.document_id,
            AiProcessingRunItem.attempt_count,
            AiProcessingRunItem.status,
        )
        .where(
            AiProcessingRunItem.status.in_(_ACTIVE),
            # `updated_at` moves on every stage transition, so a worker that is actually
            # running the pipeline keeps its item out of this window.
            AiProcessingRunItem.updated_at < cutoff,
        )
        .order_by(AiProcessingRunItem.updated_at)
        .limit(_BATCH)
        .with_for_update(skip_locked=True)
    ).all()

    if not candidates:
        session.rollback()
        return 0

    exhausted = [row for row in candidates if row.attempt_count >= settings.max_attempts]
    retryable = [row for row in candidates if row.attempt_count < settings.max_attempts]

    for row in exhausted:
        session.execute(
            update(AiProcessingRunItem)
            .where(AiProcessingRunItem.id == row.id)
            .values(
                status=RunItemStatus.FAILED.value,
                last_error_code="stale_item_abandoned",
                last_error_message=(
                    f"Stopped in '{row.status}' with no message in flight, "
                    f"after {row.attempt_count} attempts"
                ),
                completed_at=datetime.now(UTC),
            )
        )

    for row in retryable:
        # Back to `queued` and one attempt spent. A worker still holding this item will
        # find its next `advance` guard matching nothing and stop cleanly, which is the
        # same way cancellation interrupts a pipeline.
        session.execute(
            update(AiProcessingRunItem)
            .where(AiProcessingRunItem.id == row.id)
            .values(
                status=RunItemStatus.QUEUED.value,
                attempt_count=row.attempt_count + 1,
            )
        )

    session.commit()

    # A filed document whose item just went terminal would otherwise read as "still
    # processing" in the app for ever. After the commit, like every other terminal path.
    for row in exhausted:
        filing.mark_content_failed(session, row.id)
        logger.warning(
            "stale_item_abandoned",
            extra={
                "item_id": str(row.id),
                "document_id": row.document_id,
                "attempts": row.attempt_count,
                "stuck_in": row.status,
            },
        )

    # Publish only after the commit, for the reason create_run does: a worker must never
    # receive an item id whose row is not yet committed.
    requeued = 0
    for row in retryable:
        try:
            publish_processing_item(
                sqs,
                settings.sqs_queue_url,
                item_id=row.id,
                run_id=row.run_id,
                document_id=row.document_id,
                attempt=row.attempt_count + 1,
            )
        except PublishError as exc:
            # Leave it. `updated_at` has just moved, so the next sweep will not pick it up
            # until the window passes again — and the attempt it just spent means this
            # cannot repeat indefinitely.
            logger.error(
                "stale_item_republish_failed",
                extra={"item_id": str(row.id), "reason": str(exc)},
            )
            continue
        requeued += 1
        logger.info(
            "stale_item_requeued",
            extra={
                "item_id": str(row.id),
                "document_id": row.document_id,
                "stuck_in": row.status,
                "attempt": row.attempt_count + 1,
            },
        )

    return requeued + len(exhausted)
