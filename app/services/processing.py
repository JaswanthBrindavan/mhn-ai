"""Run-item state machine for the worker side.

Every transition here is guarded and idempotent, because SQS is at-least-once: the
same message may be delivered more than once, and a crashed worker's item must be
safely reclaimable. Two rules make that safe:

* **Claim under a row lock.** ``SELECT ... FOR UPDATE`` serialises concurrent claims
  of the same item, so two workers cannot both start it.
* **Advance only from the expected state.** Each transition is a conditional UPDATE
  (``WHERE status = <expected>``). If it touches zero rows the item was cancelled or
  moved by someone else, and the caller stops — this is how cancellation interrupts a
  running pipeline without the worker overwriting the ``cancelled`` state.

On redelivery the item is re-claimed from whatever state it was left in and the
pipeline restarts from the top, so each stage must be idempotent (steps 6-9 upsert
their results rather than appending).
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, delete, insert, select, update
from sqlalchemy.orm import Session

from app.models.enums import TERMINAL_STATUSES, RunItemStatus
from app.models.processing import AiProcessingRunItem
from app.models.spring import reports, unclassified_files

logger = logging.getLogger(__name__)

_TERMINAL = {status.value for status in TERMINAL_STATUSES}
# Non-terminal states a claim may start or resume from.
_CLAIMABLE = {
    RunItemStatus.PENDING.value,
    RunItemStatus.QUEUED.value,
    RunItemStatus.PROCESSING.value,
    RunItemStatus.CLASSIFYING.value,
    RunItemStatus.EXTRACTING.value,
    RunItemStatus.GENERATING_INSIGHTS.value,
}


class ClaimOutcome(StrEnum):
    PROCEED = "proceed"
    #: Already completed, rejected, failed, or cancelled — nothing to do.
    SKIP_TERMINAL = "skip_terminal"
    #: The item row is gone (e.g. its run was deleted). Message is stale.
    NOT_FOUND = "not_found"
    #: Retries exhausted; the item was marked failed by this call.
    GAVE_UP = "gave_up"


@dataclass(frozen=True)
class Claim:
    outcome: ClaimOutcome
    run_id: UUID | None = None
    document_id: int | None = None
    attempt: int = 0


def _now() -> datetime:
    return datetime.now(UTC)


def _execute_update(session: Session, stmt: Any) -> int:
    """Run a guarded UPDATE and return the affected row count, committed."""
    rows = _rowcount(session, stmt)
    session.commit()
    return rows


def _rowcount(session: Session, stmt: Any) -> int:
    """Execute a statement and return affected rows WITHOUT committing.

    Used inside a multi-statement transaction (the move) that must commit as a unit.
    """
    result = cast("CursorResult[Any]", session.execute(stmt))
    return result.rowcount


def claim_item(session: Session, item_id: UUID, *, max_attempts: int) -> Claim:
    """Take ownership of an item for one processing attempt.

    Runs in its own short transaction holding a row lock, so it never overlaps the
    long pipeline that follows. Increments the attempt counter and, once attempts are
    exhausted, marks the item failed instead of proceeding.
    """
    item = session.execute(
        select(
            AiProcessingRunItem.status,
            AiProcessingRunItem.run_id,
            AiProcessingRunItem.document_id,
            AiProcessingRunItem.attempt_count,
            AiProcessingRunItem.started_at,
        )
        .where(AiProcessingRunItem.id == item_id)
        .with_for_update()
    ).one_or_none()

    if item is None:
        session.rollback()
        return Claim(ClaimOutcome.NOT_FOUND)

    status = item.status
    if status in _TERMINAL:
        session.rollback()
        return Claim(ClaimOutcome.SKIP_TERMINAL, run_id=item.run_id, document_id=item.document_id)

    if item.attempt_count >= max_attempts:
        # Out of retries. Record a terminal failure so the message can be dropped
        # rather than redelivered forever.
        session.execute(
            update(AiProcessingRunItem)
            .where(AiProcessingRunItem.id == item_id)
            .values(
                status=RunItemStatus.FAILED.value,
                last_error_code="max_attempts_exceeded",
                last_error_message=f"Gave up after {item.attempt_count} attempts",
                completed_at=_now(),
            )
        )
        session.commit()
        logger.warning(
            "item_gave_up",
            extra={"item_id": str(item_id), "attempts": item.attempt_count},
        )
        return Claim(ClaimOutcome.GAVE_UP, run_id=item.run_id, document_id=item.document_id)

    attempt = item.attempt_count + 1
    session.execute(
        update(AiProcessingRunItem)
        .where(AiProcessingRunItem.id == item_id)
        .values(
            status=RunItemStatus.PROCESSING.value,
            attempt_count=attempt,
            started_at=item.started_at or _now(),
            # Clear a prior attempt's transient error so a later success looks clean.
            last_error_code=None,
            last_error_message=None,
        )
    )
    session.commit()
    return Claim(
        ClaimOutcome.PROCEED,
        run_id=item.run_id,
        document_id=item.document_id,
        attempt=attempt,
    )


def is_cancelled(session: Session, item_id: UUID) -> bool:
    """Cheap, lock-free read used between stages to notice a cancellation."""
    status = session.execute(
        select(AiProcessingRunItem.status).where(AiProcessingRunItem.id == item_id)
    ).scalar_one_or_none()
    # A vanished row reads as cancelled: either way, stop working on it.
    return status is None or status == RunItemStatus.CANCELLED.value


def advance(
    session: Session, item_id: UUID, *, to_status: RunItemStatus, expected: set[str]
) -> bool:
    """Move an item to the next stage only if it is still where we expect it.

    Returns False when the guarded update matches nothing — the item was cancelled or
    changed underneath us, and the caller must stop.
    """
    stmt = (
        update(AiProcessingRunItem)
        .where(
            AiProcessingRunItem.id == item_id,
            AiProcessingRunItem.status.in_(expected),
        )
        .values(status=to_status.value)
    )
    return _execute_update(session, stmt) == 1


def move_and_complete(
    session: Session,
    item_id: UUID,
    document_id: int,
    content: dict[str, Any],
    *,
    expected: set[str],
) -> bool:
    """Atomically move a classified report into ``reports`` and complete the item.

    In ONE transaction: read the source document's fields, INSERT a ``reports`` row with
    the assembled ``content``, record ``section_row_id`` and mark the item completed (guarded
    on ``expected``, so a concurrent cancel wins), then DELETE the source
    ``unclassified_files`` row. Returns False when the guard matches nothing — the whole
    transaction rolls back, so no ``reports`` row is left orphaned and no source row is
    deleted.

    Because the move and the completion commit together, a document is never in both
    tables or in neither, and a redelivery only ever sees a fully-completed item (skipped
    at claim time) or an untouched source to reprocess — never a half-done move.
    """
    src = session.execute(
        select(
            unclassified_files.c.user_id,
            unclassified_files.c.filepath,
            unclassified_files.c.private,
            unclassified_files.c.created_by,
        ).where(unclassified_files.c.id == document_id)
    ).one_or_none()

    if src is None:
        # No source to move. Under the one-active-item-per-document invariant this is an
        # anomaly (the source vanished without this item completing). Don't fabricate a
        # reports row; leave the item as-is for the guard-failure path to handle.
        session.rollback()
        logger.warning(
            "move_source_missing", extra={"item_id": str(item_id), "document_id": document_id}
        )
        return False

    reports_id = session.execute(
        insert(reports)
        .values(
            user_id=src.user_id,
            filepath=src.filepath,
            private=src.private,
            created_by=src.created_by,
            content=content,
        )
        .returning(reports.c.id)
    ).scalar_one()

    moved = _rowcount(
        session,
        update(AiProcessingRunItem)
        .where(AiProcessingRunItem.id == item_id, AiProcessingRunItem.status.in_(expected))
        .values(
            status=RunItemStatus.COMPLETED.value,
            completed_at=_now(),
            section_row_id=reports_id,
        ),
    )
    if moved != 1:
        # Cancelled or moved underneath us: undo the reports insert entirely.
        session.rollback()
        return False

    session.execute(delete(unclassified_files).where(unclassified_files.c.id == document_id))
    session.commit()
    logger.info(
        "item_moved_and_completed",
        extra={"item_id": str(item_id), "reports_id": reports_id},
    )
    return True


def complete_item(session: Session, item_id: UUID, *, expected: set[str]) -> bool:
    """Complete an item whose work is done but which is NOT moved out of intake.

    A non-report section (insurance, scans/imaging, vaccinations) is transcribed into
    ``ai_section_extractions`` and stops there: the document stays in
    ``unclassified_files`` and no section row is created. Filing it is a separate,
    undecided question — see ``docs/document-filing-design.md`` — and doing it here would
    duplicate a mover Spring already has, with a different S3 key convention.

    Guarded on ``expected`` like every other transition, so a concurrent cancel wins.
    A report never comes through here; it completes inside ``move_and_complete`` so the
    move and the completion commit together.
    """
    completed = _rowcount(
        session,
        update(AiProcessingRunItem)
        .where(AiProcessingRunItem.id == item_id, AiProcessingRunItem.status.in_(expected))
        .values(status=RunItemStatus.COMPLETED.value, completed_at=_now()),
    )
    if completed != 1:
        session.rollback()
        return False
    session.commit()
    logger.info("item_completed_in_place", extra={"item_id": str(item_id)})
    return True


def reject_item(
    session: Session, item_id: UUID, *, code: str, message: str, expected: set[str]
) -> bool:
    """Terminally reject an item (wrong document type, unprocessable content)."""
    stmt = (
        update(AiProcessingRunItem)
        .where(
            AiProcessingRunItem.id == item_id,
            AiProcessingRunItem.status.in_(expected),
        )
        .values(
            status=RunItemStatus.REJECTED.value,
            last_error_code=code,
            last_error_message=message,
            completed_at=_now(),
        )
    )
    return _execute_update(session, stmt) == 1


CLAIMABLE_STATUSES = _CLAIMABLE
