"""Assembling a filed document's `content` from the per-stage results, and closing the
item afterwards. The filing move itself lives in test_filing.py.
"""

import json
import uuid

import pytest
from sqlalchemy import text

from app.services import processing
from app.services.assembly import CONTENT_SCHEMA_VERSION, ContentState, build_content

pytestmark = pytest.mark.integration

_IN_PROGRESS = {"processing", "classifying", "extracting", "generating_insights"}


def _seed_item(db_session, document_id, status="generating_insights") -> uuid.UUID:
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status) "
            "VALUES (:r, :d, :s) RETURNING id"
        ),
        {"r": run_id, "d": document_id, "s": status},
    ).scalar_one()
    db_session.flush()
    return item_id


def _seed_classification(db_session, item_id, document_id, section="reports") -> None:
    db_session.execute(
        text(
            "INSERT INTO ai_report_classifications (run_item_id, document_id, section, title, "
            "confidence, prompt_version, schema_version) "
            "VALUES (:i, :d, :s, 'Complete Blood Count', 0.97, 'clf-2', 'clf-2')"
        ),
        {"i": item_id, "d": document_id, "s": section},
    )
    db_session.flush()


def _seed_section_extraction(db_session, item_id, document_id) -> None:
    """A non-report section's transcription, as section_extraction.extract_section writes it."""
    db_session.execute(
        text(
            "INSERT INTO ai_section_extractions "
            "(run_item_id, document_id, section, data, prompt_version, schema_version) "
            "VALUES (:i, :d, 'insurance', CAST(:data AS JSONB), 'sec-1', 'sec-1')"
        ),
        {
            "i": item_id,
            "d": document_id,
            "data": json.dumps(
                {
                    "section": "insurance",
                    "fields": {"insurer": "Star Health", "start_date": "2019-10-01"},
                    "flags": [],
                }
            ),
        },
    )
    db_session.flush()


def _seed_stage_rows(db_session, item_id, document_id) -> None:
    _seed_classification(db_session, item_id, document_id)
    db_session.execute(
        text(
            "INSERT INTO ai_report_extractions "
            "(run_item_id, document_id, data, prompt_version, schema_version) "
            "VALUES (:i, :d, CAST(:data AS JSONB), 'ext-1', 'ext-1')"
        ),
        {
            "i": item_id,
            "d": document_id,
            "data": json.dumps({"results": [{"test_name": "Glucose", "abnormal_flag": "high"}]}),
        },
    )
    db_session.execute(
        text(
            "INSERT INTO ai_report_insights "
            "(run_item_id, document_id, data, prompt_version, schema_version) "
            "VALUES (:i, :d, CAST(:data AS JSONB), 'ins-1', 'ins-1')"
        ),
        {
            "i": item_id,
            "d": document_id,
            "data": json.dumps({"insights": [], "summary": None, "disclaimer": "info only"}),
        },
    )
    db_session.flush()


def _item(db_session, item_id):
    return (
        db_session.execute(
            text(
                "SELECT status, section_row_id, completed_at "
                "FROM ai_processing_run_items WHERE id=:id"
            ),
            {"id": item_id},
        )
        .mappings()
        .one()
    )


# --- build_content ----------------------------------------------------------


def test_build_content_assembles_all_three_stages(db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    _seed_stage_rows(db_session, item_id, document_id)

    content = build_content(db_session, item_id, state=ContentState.COMPLETE)

    ai = content["ai"]
    assert ai["schema_version"] == CONTENT_SCHEMA_VERSION
    assert ai["classification"]["title"] == "Complete Blood Count"
    assert ai["classification"]["confidence"] == 0.97  # Decimal -> float
    assert ai["extraction"]["results"][0]["abnormal_flag"] == "high"
    assert ai["insights"]["disclaimer"] == "info only"
    assert "generated_at" in ai


def test_content_at_filing_time_carries_only_the_classification(db_session, make_document):
    """Written the moment the document is filed, before any extraction has run."""
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, status="classifying")
    _seed_classification(db_session, item_id, document_id)

    content = build_content(db_session, item_id, state=ContentState.CLASSIFIED)["ai"]

    assert content["state"] == "classified"
    assert content["schema_version"] == "2.1"
    assert content["classification"]["section"] == "reports"
    assert content["extraction"] is None
    assert content["section_extraction"] is None
    assert content["insights"] is None
    # The intake id, carried because filing DELETES the row it names. It is the only
    # identifier `/refile` and `/status` accept, and once the document is filed the app
    # reaches it through its section row — so without this the payload is the last place
    # the two ids are connected, and the mismatch flag would offer an action nothing
    # could address.
    assert content["document_id"] == document_id


def test_completed_report_content_has_extraction_and_insights(db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    _seed_stage_rows(db_session, item_id, document_id)

    content = build_content(db_session, item_id, state=ContentState.COMPLETE)["ai"]

    assert content["state"] == "complete"
    assert content["extraction"]["results"]
    assert content["insights"]["disclaimer"]
    # A report never writes the section-extraction shape.
    assert content["section_extraction"] is None


def test_completed_section_content_has_section_extraction_only(db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, status="extracting")
    _seed_classification(db_session, item_id, document_id, section="insurance")
    _seed_section_extraction(db_session, item_id, document_id)

    content = build_content(db_session, item_id, state=ContentState.COMPLETE)["ai"]

    assert content["state"] == "complete"
    assert content["section_extraction"]["fields"]
    # Mutually exclusive: the two carry different shapes and must never both be populated.
    assert content["extraction"] is None
    assert content["insights"] is None


# --- completing the item ----------------------------------------------------
# Filing itself (INSERT the section row, record section_row_id, DELETE the intake row) is
# `app.services.filing` and is covered by test_filing.py. What is left here is the item's
# own completion, which happens after the filed row's content has been updated.


def test_complete_item_closes_the_item_after_its_content_was_written(db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    _seed_stage_rows(db_session, item_id, document_id)

    assert processing.complete_item(db_session, item_id, expected=_IN_PROGRESS) is True

    row = _item(db_session, item_id)
    assert row["status"] == "completed"
    assert row["completed_at"] is not None


def test_complete_item_is_refused_when_the_item_was_cancelled(db_session, make_document):
    """The guard is how a cancel interrupts a running pipeline: the worker must not
    overwrite `cancelled` with `completed`."""
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, status="cancelled")
    # Persist the seed past the savepoint complete_item rolls back to on refusal.
    db_session.commit()

    assert processing.complete_item(db_session, item_id, expected=_IN_PROGRESS) is False
    assert _item(db_session, item_id)["status"] == "cancelled"
