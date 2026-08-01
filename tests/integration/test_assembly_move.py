"""The assemble & move capstone: build reports.content from the stage results, INSERT the
reports row, record section_row_id, DELETE unclassified_files — atomically and guarded.
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


def _reports_row(db_session, reports_id):
    return (
        db_session.execute(text("SELECT * FROM reports WHERE id = :id"), {"id": reports_id})
        .mappings()
        .one_or_none()
    )


def _unclassified_exists(db_session, document_id) -> bool:
    return (
        db_session.execute(
            text("SELECT 1 FROM unclassified_files WHERE id = :id"), {"id": document_id}
        ).scalar_one_or_none()
        is not None
    )


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
    assert content["schema_version"] == "2.0"
    assert content["classification"]["section"] == "reports"
    assert content["extraction"] is None
    assert content["section_extraction"] is None
    assert content["insights"] is None


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


# --- move_and_complete ------------------------------------------------------


def test_move_creates_report_records_id_and_deletes_source(db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    _seed_stage_rows(db_session, item_id, document_id)
    content = build_content(db_session, item_id, state=ContentState.COMPLETE)

    ok = processing.move_and_complete(
        db_session, item_id, document_id, content, expected=_IN_PROGRESS
    )

    assert ok is True
    row = _item(db_session, item_id)
    assert row["status"] == "completed"
    assert row["section_row_id"] is not None
    assert row["completed_at"] is not None

    report = _reports_row(db_session, row["section_row_id"])
    assert report is not None
    assert report["content"]["ai"]["classification"]["section"] == "reports"
    # The report carries the source document's fields; the source row is gone.
    assert report["filepath"]  # copied from unclassified_files
    assert _unclassified_exists(db_session, document_id) is False


def test_move_is_rolled_back_when_cancelled(db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, status="cancelled")
    _seed_stage_rows(db_session, item_id, document_id)
    content = build_content(db_session, item_id, state=ContentState.COMPLETE)
    src_key = db_session.execute(
        text("SELECT filepath FROM unclassified_files WHERE id=:id"), {"id": document_id}
    ).scalar_one()
    # Persist the seed past the savepoint move_and_complete will roll back to.
    db_session.commit()

    ok = processing.move_and_complete(
        db_session, item_id, document_id, content, expected=_IN_PROGRESS
    )

    assert ok is False
    # No reports row was left behind, the source survives, and the cancel stands.
    assert _item(db_session, item_id)["status"] == "cancelled"
    assert _unclassified_exists(db_session, document_id) is True
    orphans = db_session.execute(
        text("SELECT count(*) FROM reports WHERE filepath = :k"), {"k": src_key}
    ).scalar_one()
    assert orphans == 0
