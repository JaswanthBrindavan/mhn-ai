"""The worker-side state machine: claim, guarded transitions, cancellation, attempts."""

import uuid

import pytest
from sqlalchemy import text

from app.models.enums import RunItemStatus
from app.services import processing
from app.services.processing import ClaimOutcome

pytestmark = pytest.mark.integration

_IN_PROGRESS = {"processing", "classifying", "extracting", "generating_insights"}


def _make_item(db_session, make_document, status: str = "queued", attempt: int = 0) -> uuid.UUID:
    document_id = make_document()
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status, attempt_count) "
            "VALUES (:r, :rep, :s, :a) RETURNING id"
        ),
        {"r": run_id, "rep": document_id, "s": status, "a": attempt},
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


def test_claim_a_queued_item_starts_processing(db_session, make_document):
    item_id = _make_item(db_session, make_document, status="queued")

    claim = processing.claim_item(db_session, item_id, max_attempts=3)

    assert claim.outcome is ClaimOutcome.PROCEED
    assert claim.attempt == 1
    assert _status(db_session, item_id) == RunItemStatus.PROCESSING.value


def test_claiming_a_completed_item_is_skipped(db_session, make_document):
    item_id = _make_item(db_session, make_document, status="completed")

    claim = processing.claim_item(db_session, item_id, max_attempts=3)

    assert claim.outcome is ClaimOutcome.SKIP_TERMINAL
    # Untouched.
    assert _status(db_session, item_id) == "completed"


def test_claiming_a_cancelled_item_is_skipped(db_session, make_document):
    item_id = _make_item(db_session, make_document, status="cancelled")
    claim = processing.claim_item(db_session, item_id, max_attempts=3)
    assert claim.outcome is ClaimOutcome.SKIP_TERMINAL


def test_claiming_a_missing_item_reports_not_found(db_session):
    claim = processing.claim_item(db_session, uuid.uuid4(), max_attempts=3)
    assert claim.outcome is ClaimOutcome.NOT_FOUND


def test_claim_gives_up_after_max_attempts(db_session, make_document):
    # Already at the cap: the next claim should fail it, not process it.
    item_id = _make_item(db_session, make_document, status="processing", attempt=3)

    claim = processing.claim_item(db_session, item_id, max_attempts=3)

    assert claim.outcome is ClaimOutcome.GAVE_UP
    assert _status(db_session, item_id) == RunItemStatus.FAILED.value
    code = db_session.execute(
        text("SELECT last_error_code FROM ai_processing_run_items WHERE id = :id"),
        {"id": item_id},
    ).scalar_one()
    assert code == "max_attempts_exceeded"


def test_claim_increments_attempt_each_time(db_session, make_document):
    item_id = _make_item(db_session, make_document, status="queued", attempt=0)
    assert processing.claim_item(db_session, item_id, max_attempts=5).attempt == 1
    # Simulate redelivery: still non-terminal, claim again.
    assert processing.claim_item(db_session, item_id, max_attempts=5).attempt == 2


# --- guarded transitions ----------------------------------------------------


def test_advance_moves_from_expected_state(db_session, make_document):
    item_id = _make_item(db_session, make_document, status="processing")

    ok = processing.advance(
        db_session, item_id, to_status=RunItemStatus.CLASSIFYING, expected=_IN_PROGRESS
    )

    assert ok is True
    assert _status(db_session, item_id) == "classifying"


def test_advance_refuses_when_cancelled(db_session, make_document):
    item_id = _make_item(db_session, make_document, status="cancelled")

    ok = processing.advance(
        db_session, item_id, to_status=RunItemStatus.CLASSIFYING, expected=_IN_PROGRESS
    )

    assert ok is False
    # The guard did not overwrite the cancellation.
    assert _status(db_session, item_id) == "cancelled"


def test_is_cancelled_detects_cancellation(db_session, make_document):
    item_id = _make_item(db_session, make_document, status="cancelled")
    assert processing.is_cancelled(db_session, item_id) is True


def test_is_cancelled_false_while_processing(db_session, make_document):
    item_id = _make_item(db_session, make_document, status="processing")
    assert processing.is_cancelled(db_session, item_id) is False


def _error_code(db_session, item_id) -> str | None:
    return db_session.execute(
        text("SELECT last_error_code FROM ai_processing_run_items WHERE id = :id"),
        {"id": item_id},
    ).scalar_one()


def test_a_retry_is_noted_without_moving_the_stage(db_session, make_document):
    # The item really is still at that stage — redelivery re-claims it there — so only
    # the error code changes. Moving the status would make the retry look like a new run.
    item_id = _make_item(db_session, make_document, status="extracting")

    noted = processing.note_retry(
        db_session, item_id, message="503 model unavailable", expected=_IN_PROGRESS
    )

    assert noted is True
    assert _status(db_session, item_id) == "extracting"
    assert _error_code(db_session, item_id) == processing.RETRYING


def test_the_next_attempt_clears_the_retry_note(db_session, make_document):
    # What makes the code mean "waiting to come back" rather than "failed": claim_item
    # wipes it the moment the work actually restarts.
    item_id = _make_item(db_session, make_document, status="extracting")
    processing.note_retry(
        db_session, item_id, message="503 model unavailable", expected=_IN_PROGRESS
    )

    processing.claim_item(db_session, item_id, max_attempts=3)

    assert _error_code(db_session, item_id) is None


def test_a_settled_item_is_not_marked_as_retrying(db_session, make_document):
    # The guard matters: a cancellation or a completion racing the failing attempt must
    # not be relabelled as work still in flight.
    item_id = _make_item(db_session, make_document, status="cancelled")

    noted = processing.note_retry(
        db_session, item_id, message="503 model unavailable", expected=_IN_PROGRESS
    )

    assert noted is False
    assert _error_code(db_session, item_id) is None
