"""The worker-side state machine: claim, guarded transitions, cancellation, attempts."""

import uuid

import pytest
from sqlalchemy import text

from app.models.enums import RunItemStatus
from app.services import processing
from app.services.processing import ClaimOutcome

pytestmark = pytest.mark.integration

_IN_PROGRESS = {"processing", "classifying", "extracting", "generating_insights"}


def _make_item(db_session, make_report, status: str = "queued", attempt: int = 0) -> uuid.UUID:
    report_id = make_report()
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, report_id, status, attempt_count) "
            "VALUES (:r, :rep, :s, :a) RETURNING id"
        ),
        {"r": run_id, "rep": report_id, "s": status, "a": attempt},
    ).scalar_one()
    # Commit (releases the savepoint) so claim_item's own rollback on skip paths —
    # correct in production where it holds a fresh session — cannot discard the seed.
    db_session.commit()
    return item_id


def _status(db_session, item_id) -> str:
    return db_session.execute(
        text("SELECT status FROM ai_processing_run_items WHERE id = :id"), {"id": item_id}
    ).scalar_one()


# --- claim ------------------------------------------------------------------


def test_claim_a_queued_item_starts_processing(db_session, make_report):
    item_id = _make_item(db_session, make_report, status="queued")

    claim = processing.claim_item(db_session, item_id, max_attempts=3)

    assert claim.outcome is ClaimOutcome.PROCEED
    assert claim.attempt == 1
    assert _status(db_session, item_id) == RunItemStatus.PROCESSING.value


def test_claiming_a_completed_item_is_skipped(db_session, make_report):
    item_id = _make_item(db_session, make_report, status="completed")

    claim = processing.claim_item(db_session, item_id, max_attempts=3)

    assert claim.outcome is ClaimOutcome.SKIP_TERMINAL
    # Untouched.
    assert _status(db_session, item_id) == "completed"


def test_claiming_a_cancelled_item_is_skipped(db_session, make_report):
    item_id = _make_item(db_session, make_report, status="cancelled")
    claim = processing.claim_item(db_session, item_id, max_attempts=3)
    assert claim.outcome is ClaimOutcome.SKIP_TERMINAL


def test_claiming_a_missing_item_reports_not_found(db_session):
    claim = processing.claim_item(db_session, uuid.uuid4(), max_attempts=3)
    assert claim.outcome is ClaimOutcome.NOT_FOUND


def test_claim_gives_up_after_max_attempts(db_session, make_report):
    # Already at the cap: the next claim should fail it, not process it.
    item_id = _make_item(db_session, make_report, status="processing", attempt=3)

    claim = processing.claim_item(db_session, item_id, max_attempts=3)

    assert claim.outcome is ClaimOutcome.GAVE_UP
    assert _status(db_session, item_id) == RunItemStatus.FAILED.value
    code = db_session.execute(
        text("SELECT last_error_code FROM ai_processing_run_items WHERE id = :id"),
        {"id": item_id},
    ).scalar_one()
    assert code == "max_attempts_exceeded"


def test_claim_increments_attempt_each_time(db_session, make_report):
    item_id = _make_item(db_session, make_report, status="queued", attempt=0)
    assert processing.claim_item(db_session, item_id, max_attempts=5).attempt == 1
    # Simulate redelivery: still non-terminal, claim again.
    assert processing.claim_item(db_session, item_id, max_attempts=5).attempt == 2


# --- guarded transitions ----------------------------------------------------


def test_advance_moves_from_expected_state(db_session, make_report):
    item_id = _make_item(db_session, make_report, status="processing")

    ok = processing.advance(
        db_session, item_id, to_status=RunItemStatus.CLASSIFYING, expected=_IN_PROGRESS
    )

    assert ok is True
    assert _status(db_session, item_id) == "classifying"


def test_advance_refuses_when_cancelled(db_session, make_report):
    item_id = _make_item(db_session, make_report, status="cancelled")

    ok = processing.advance(
        db_session, item_id, to_status=RunItemStatus.CLASSIFYING, expected=_IN_PROGRESS
    )

    assert ok is False
    # The guard did not overwrite the cancellation.
    assert _status(db_session, item_id) == "cancelled"


def test_complete_is_guarded_against_cancellation(db_session, make_report):
    item_id = _make_item(db_session, make_report, status="cancelled")
    ok = processing.complete_item(db_session, item_id, expected=_IN_PROGRESS)
    assert ok is False
    assert _status(db_session, item_id) == "cancelled"


def test_is_cancelled_detects_cancellation(db_session, make_report):
    item_id = _make_item(db_session, make_report, status="cancelled")
    assert processing.is_cancelled(db_session, item_id) is True


def test_is_cancelled_false_while_processing(db_session, make_report):
    item_id = _make_item(db_session, make_report, status="processing")
    assert processing.is_cancelled(db_session, item_id) is False
