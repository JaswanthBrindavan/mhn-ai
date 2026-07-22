"""Report-processing run business logic.

Idempotency is enforced by the database, not by application checks: a partial unique
index permits at most one in-flight item per report. Two concurrent submissions for the
same report therefore cannot both create an item — one raises ``IntegrityError`` and we
reuse the winner's row. A read-then-write check alone would let a race through.
"""

import uuid
from collections import Counter

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.errors import ApiError
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

_ACTIVE = {status.value for status in ACTIVE_STATUSES}
_CANCELLABLE = {status.value for status in CANCELLABLE_STATUSES}


def _existing_report_ids(session: Session, report_ids: list[int]) -> set[int]:
    """Which of these reports exist. A sanity check, NOT an access-control check."""
    rows = session.execute(select(reports.c.id).where(reports.c.id.in_(report_ids))).scalars()
    return set(rows)


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
    session: Session, payload: CreateRunRequest, request_id: str | None
) -> CreateRunResponse:
    """Persist a run and its items. Publishing to SQS arrives in step 4."""
    # Deduplicate while preserving caller order, so a repeated id in one request
    # cannot try to create two items and trip the unique index against itself.
    unique_ids = list(dict.fromkeys(payload.report_ids))

    existing = _existing_report_ids(session, unique_ids)
    missing = [report_id for report_id in unique_ids if report_id not in existing]
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

    submitted: list[SubmittedItem] = []
    for report_id in unique_ids:
        submitted.append(_submit_one(session, run, report_id, payload.force_reprocess))

    session.commit()
    session.refresh(run)

    return CreateRunResponse(run_id=run.id, created_at=run.created_at, items=submitted)


def _submit_one(
    session: Session,
    run: AiProcessingRun,
    report_id: int,
    force_reprocess: bool,
) -> SubmittedItem:
    """Create, or safely reuse, the item for one report."""
    active = _active_item_for_report(session, report_id)
    if active is not None:
        # Already in flight. Reuse rather than process the same report twice --
        # force_reprocess does not apply, since the work has not finished yet.
        return SubmittedItem(
            report_id=report_id,
            item_id=active.id,
            status=active.status,
            outcome=SubmitOutcome.REUSED,
        )

    latest = _latest_item_for_report(session, report_id)
    if (
        latest is not None
        and latest.status == RunItemStatus.COMPLETED.value
        and not force_reprocess
    ):
        # Never overwrite a completed result without an explicit force_reprocess.
        return SubmittedItem(
            report_id=report_id,
            item_id=latest.id,
            status=latest.status,
            outcome=SubmitOutcome.ALREADY_COMPLETED,
        )

    item = AiProcessingRunItem(
        run_id=run.id,
        report_id=report_id,
        status=RunItemStatus.PENDING.value,
    )
    session.add(item)
    try:
        session.flush()
    except IntegrityError:
        # Lost a race with a concurrent submission. The unique index did its job;
        # roll back to the savepoint and reuse whatever the winner created.
        session.rollback()
        session.add(run)
        winner = _active_item_for_report(session, report_id)
        if winner is None:  # pragma: no cover - only on an unrelated constraint
            raise
        return SubmittedItem(
            report_id=report_id,
            item_id=winner.id,
            status=winner.status,
            outcome=SubmitOutcome.REUSED,
        )

    return SubmittedItem(
        report_id=report_id,
        item_id=item.id,
        status=item.status,
        outcome=SubmitOutcome.CREATED,
    )


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
