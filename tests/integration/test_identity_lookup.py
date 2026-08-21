"""Resolving who a document is for, and reading back the verdict reached about it."""

import uuid

import pytest
from sqlalchemy import text

from app.services.identity import confirm_identity, owner_name, record_verdict, settled_verdict
from app.services.names import NameVerdict

pytestmark = pytest.mark.integration


def _rename_owner(db_session, document_id: int, name: str) -> None:
    db_session.execute(
        text(
            'UPDATE "user" SET name = :n WHERE id = '
            "(SELECT user_id FROM unclassified_files WHERE id = :d)"
        ),
        {"n": name, "d": document_id},
    )


def _classify(db_session, document_id: int, *, name_match=None, created_at=None) -> uuid.UUID:
    """A finished run item with its classification row. Terminal status on purpose: two
    ACTIVE items for one document would trip uq_ai_run_items_active_document."""
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status) "
            "VALUES (:r, :d, 'completed') RETURNING id"
        ),
        {"r": run_id, "d": document_id},
    ).scalar_one()
    db_session.execute(
        text(
            "INSERT INTO ai_report_classifications "
            "(run_item_id, document_id, section, title, confidence, patient_name, name_match, "
            " prompt_version, schema_version, created_at) "
            "VALUES (:i, :d, 'reports', 'A Report', 0.9, 'Rajesh Kumar Sharma', :m, "
            "'clf-1', 'clf-1', COALESCE(CAST(:c AS timestamptz), now()))"
        ),
        {"i": item_id, "d": document_id, "m": name_match, "c": created_at},
    )
    db_session.commit()
    return uuid.UUID(str(item_id))


def test_owner_name_comes_from_the_intake_row(db_session, make_document):
    document_id = make_document()
    _rename_owner(db_session, document_id, "Rajesh Kumar Sharma")
    db_session.commit()

    assert owner_name(db_session, document_id) == "Rajesh Kumar Sharma"


def test_owner_name_is_none_when_the_intake_row_is_gone(db_session, make_document):
    """Filing deletes the intake row. The gate must not fail over that."""
    document_id = make_document()
    db_session.execute(text("DELETE FROM unclassified_files WHERE id = :d"), {"d": document_id})
    db_session.commit()

    assert owner_name(db_session, document_id) is None
    assert owner_name(db_session, 999_999_999) is None


def test_settled_verdict_is_none_before_anything_is_decided(db_session, make_document):
    document_id = make_document()
    db_session.commit()
    assert settled_verdict(db_session, document_id) is None

    # A classification exists, but the gate has not run: still nothing settled.
    _classify(db_session, document_id)
    assert settled_verdict(db_session, document_id) is None


def test_record_verdict_is_read_back_by_document(db_session, make_document):
    document_id = make_document()
    item_id = _classify(db_session, document_id)

    record_verdict(db_session, item_id, NameVerdict.MISMATCH)

    assert settled_verdict(db_session, document_id) is NameVerdict.MISMATCH


def test_a_confirmed_identity_outranks_the_computed_mismatch(db_session, make_document):
    document_id = make_document()
    item_id = _classify(db_session, document_id)
    record_verdict(db_session, item_id, NameVerdict.MISMATCH)

    assert confirm_identity(db_session, document_id) is True

    assert settled_verdict(db_session, document_id) is NameVerdict.MATCH


def test_the_verdict_outlives_the_run_item_that_produced_it(db_session, make_document):
    """A retry mints a NEW run item; the decision is keyed on the document, not the item.

    created_at is set explicitly because `now()` is the transaction's timestamp, so two
    rows written in one test would otherwise tie.
    """
    document_id = make_document()
    _classify(db_session, document_id, name_match="mismatch", created_at="2026-08-20 10:00+00")
    retry_item = _classify(db_session, document_id, created_at="2026-08-21 10:00+00")

    # The newest classification has no verdict yet, so nothing is settled...
    assert settled_verdict(db_session, document_id) is None
    # ...until the gate records one against the new item.
    record_verdict(db_session, retry_item, NameVerdict.MATCH)
    assert settled_verdict(db_session, document_id) is NameVerdict.MATCH


def test_confirm_identity_is_false_when_there_is_nothing_to_stamp(db_session, make_document):
    document_id = make_document()
    db_session.commit()

    assert confirm_identity(db_session, document_id) is False
