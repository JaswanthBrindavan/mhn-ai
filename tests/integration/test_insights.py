"""The insights stage: reads the extraction, generates informational insights over the
structured data (not the raw file), persists with a disclaimer, and logs cost.
"""

import json
import uuid

import pytest
from sqlalchemy import text

from app.services.insights import ALL_IN_RANGE_SUMMARY, DISCLAIMER, generate_insights
from app.workers.stagetypes import StageContext, TransientStageError
from tests.support.ai import FakeAIProvider, structured_response

pytestmark = pytest.mark.integration


def _seed_item(db_session, document_id: int = 4242) -> uuid.UUID:
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status) "
            "VALUES (:r, :rep, 'generating_insights') RETURNING id"
        ),
        {"r": run_id, "rep": document_id},
    ).scalar_one()
    db_session.flush()
    return item_id


def _seed_extraction(db_session, item_id, document_id, data: dict) -> None:
    db_session.execute(
        text(
            "INSERT INTO ai_report_extractions "
            "(run_item_id, document_id, data, prompt_version, schema_version) "
            "VALUES (:i, :d, CAST(:data AS JSONB), 'ext-1', 'ext-1')"
        ),
        {"i": item_id, "d": document_id, "data": json.dumps(data)},
    )
    db_session.flush()


def _extraction_data(**over):
    data = {
        "results": [
            {
                "test_name": "Fasting Glucose",
                "value": "126",
                "unit": "mg/dL",
                "reference_range": "70-99",
                "abnormal_flag": "high",
                "value_numeric": 126.0,
            }
        ],
        "report_date": "2026-07-20",
    }
    data.update(over)
    return data


def _context(db_session, test_settings, document_id, item_id, ai, attempt=1) -> StageContext:
    return StageContext(
        item_id=item_id,
        run_id=uuid.uuid4(),
        document_id=document_id,
        attempt=attempt,
        session=db_session,
        s3=None,  # insights never touches S3
        ai=ai,
        settings=test_settings,
    )


def _insights_row(db_session, item_id):
    return (
        db_session.execute(
            text("SELECT * FROM ai_report_insights WHERE run_item_id = :id"), {"id": item_id}
        )
        .mappings()
        .one_or_none()
    )


def _logs(db_session, item_id):
    return (
        db_session.execute(
            text("SELECT * FROM ai_process_logs WHERE run_item_id = :id ORDER BY attempt"),
            {"id": item_id},
        )
        .mappings()
        .all()
    )


# --- happy path -------------------------------------------------------------


def test_insights_persist_with_disclaimer_and_reason_over_extracted_data(db_session, test_settings):
    document_id = 100
    item_id = _seed_item(db_session, document_id)
    _seed_extraction(db_session, item_id, document_id, _extraction_data())
    ai = FakeAIProvider()  # default insights payload

    generate_insights(_context(db_session, test_settings, document_id, item_id, ai))

    row = _insights_row(db_session, item_id)
    assert row is not None
    assert row["data"]["disclaimer"] == DISCLAIMER
    assert row["data"]["insights"][0]["related_tests"] == ["Fasting Glucose"]

    # The model was given our deterministic flag and the test name, and NO document.
    assert ai.calls[0]["document"] is None
    assert "high" in ai.last_instruction
    assert "Fasting Glucose" in ai.last_instruction

    assert _logs(db_session, item_id)[0]["outcome"] == "succeeded"


def test_no_extracted_results_skips_the_model_call(db_session, test_settings):
    document_id = 101
    item_id = _seed_item(db_session, document_id)
    _seed_extraction(db_session, item_id, document_id, _extraction_data(results=[]))
    ai = FakeAIProvider()

    generate_insights(_context(db_session, test_settings, document_id, item_id, ai))

    # Nothing to interpret: no paid call, but a disclaimered empty payload is stored.
    assert ai.calls == []
    row = _insights_row(db_session, item_id)
    assert row["data"]["insights"] == []
    assert row["data"]["disclaimer"] == DISCLAIMER
    log = _logs(db_session, item_id)[0]
    assert log["outcome"] == "succeeded"
    assert log["input_tokens"] == 0  # no model call


def test_all_results_in_range_skips_the_model_call(db_session, test_settings):
    document_id = 103
    item_id = _seed_item(db_session, document_id)
    normal = [
        dict(_extraction_data()["results"][0], abnormal_flag="normal", value="88"),
        dict(_extraction_data()["results"][0], test_name="Sodium", abnormal_flag="normal"),
    ]
    _seed_extraction(db_session, item_id, document_id, _extraction_data(results=normal))
    ai = FakeAIProvider()

    generate_insights(_context(db_session, test_settings, document_id, item_id, ai))

    # Nothing noteworthy: no paid call, but the report is still recorded as checked.
    assert ai.calls == []
    row = _insights_row(db_session, item_id)
    assert row["data"]["insights"] == []
    assert row["data"]["summary"] == ALL_IN_RANGE_SUMMARY
    assert row["data"]["disclaimer"] == DISCLAIMER
    assert _logs(db_session, item_id)[0]["outcome"] == "succeeded"


def test_undetermined_flag_still_calls_the_model(db_session, test_settings):
    """None means 'could not be checked', which is not the same as in range."""
    document_id = 104
    item_id = _seed_item(db_session, document_id)
    unchecked = [
        dict(_extraction_data()["results"][0], abnormal_flag="normal"),
        dict(_extraction_data()["results"][0], test_name="Culture", abnormal_flag=None),
    ]
    _seed_extraction(db_session, item_id, document_id, _extraction_data(results=unchecked))
    ai = FakeAIProvider()

    generate_insights(_context(db_session, test_settings, document_id, item_id, ai))

    assert len(ai.calls) == 1


# --- failure handling -------------------------------------------------------


def test_missing_extraction_is_transient(db_session, test_settings):
    document_id = 102
    item_id = _seed_item(db_session, document_id)  # no extraction row seeded

    ctx = _context(db_session, test_settings, document_id, item_id, FakeAIProvider())
    with pytest.raises(TransientStageError):
        generate_insights(ctx)

    assert _insights_row(db_session, item_id) is None


def test_invalid_model_output_is_transient_and_not_persisted(db_session, test_settings):
    document_id = 103
    item_id = _seed_item(db_session, document_id)
    _seed_extraction(db_session, item_id, document_id, _extraction_data())
    ai = FakeAIProvider(response=structured_response('{"insights": [{"heading":'))  # broken JSON

    with pytest.raises(TransientStageError):
        generate_insights(_context(db_session, test_settings, document_id, item_id, ai))

    assert _insights_row(db_session, item_id) is None
    assert _logs(db_session, item_id)[0]["outcome"] == "validation_failed"


# --- idempotency ------------------------------------------------------------


def test_rerun_same_attempt_updates_rather_than_duplicates(db_session, test_settings):
    document_id = 104
    item_id = _seed_item(db_session, document_id)
    _seed_extraction(db_session, item_id, document_id, _extraction_data())
    ctx = _context(db_session, test_settings, document_id, item_id, FakeAIProvider())

    generate_insights(ctx)
    generate_insights(ctx)

    assert len(_logs(db_session, item_id)) == 1
    count = db_session.execute(
        text("SELECT count(*) FROM ai_report_insights WHERE run_item_id = :id"),
        {"id": item_id},
    ).scalar_one()
    assert count == 1
