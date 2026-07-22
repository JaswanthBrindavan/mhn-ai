"""The database-level idempotency guarantee.

The partial unique index is what actually prevents duplicate processing. Application
checks alone lose to a race; these tests exercise the constraint directly.
"""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.models.enums import ACTIVE_STATUSES, TERMINAL_STATUSES

pytestmark = pytest.mark.integration


def _insert_item(db_session, report_id: int, status: str) -> uuid.UUID:
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    return db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, report_id, status) "
            "VALUES (:run_id, :report_id, :status) RETURNING id"
        ),
        {"run_id": run_id, "report_id": report_id, "status": status},
    ).scalar_one()


@pytest.mark.parametrize("status", sorted(s.value for s in ACTIVE_STATUSES))
def test_two_active_items_for_one_report_are_impossible(db_session, make_report, status):
    report_id = make_report()
    _insert_item(db_session, report_id, status)
    db_session.flush()

    with pytest.raises(IntegrityError):
        _insert_item(db_session, report_id, status)
        db_session.flush()


@pytest.mark.parametrize("status", sorted(s.value for s in TERMINAL_STATUSES))
def test_terminal_items_do_not_block_new_work(db_session, make_report, status):
    report_id = make_report()
    _insert_item(db_session, report_id, status)
    db_session.flush()

    # A finished item must not prevent a retry or a forced reprocess.
    _insert_item(db_session, report_id, "pending")
    db_session.flush()


def test_many_terminal_items_may_coexist(db_session, make_report):
    """Retries accumulate history; only the in-flight one is unique."""
    report_id = make_report()
    for status in ("failed", "failed", "cancelled", "completed"):
        _insert_item(db_session, report_id, status)
    db_session.flush()


def test_status_check_constraint_rejects_unknown_states(db_session, make_report):
    report_id = make_report()
    with pytest.raises(IntegrityError):
        _insert_item(db_session, report_id, "not_a_real_status")
        db_session.flush()
