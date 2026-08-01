"""The extraction stage: persistence, deterministic normalisation, failure handling, cost
logging. Runs against the live DB and moto S3 with a fake AI provider.
"""

import uuid

import pytest
from sqlalchemy import text

from app.services import extraction
from app.services.extraction import extract_report
from app.workers.stagetypes import StageContext, TransientStageError
from tests.integration.conftest import document_key
from tests.support.ai import FakeAIProvider, extraction_payload, structured_response

pytestmark = pytest.mark.integration


def _seed_item(db_session, document_id: int) -> uuid.UUID:
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status) "
            "VALUES (:r, :rep, 'extracting') RETURNING id"
        ),
        {"r": run_id, "rep": document_id},
    ).scalar_one()
    db_session.flush()
    return item_id


def _context(db_session, aws, test_settings, document_id, item_id, ai, attempt=1) -> StageContext:
    return StageContext(
        item_id=item_id,
        run_id=uuid.uuid4(),
        document_id=document_id,
        source_key=document_key(db_session, document_id),
        attempt=attempt,
        session=db_session,
        s3=aws[0],
        ai=ai,
        settings=test_settings,
    )


def _extraction_row(db_session, item_id):
    return (
        db_session.execute(
            text("SELECT * FROM ai_report_extractions WHERE run_item_id = :id"),
            {"id": item_id},
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


# --- happy path + deterministic normalisation -------------------------------


def test_extraction_persists_result_and_normalises_deterministically(
    db_session, make_document, aws, test_settings
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider()  # default extraction payload: glucose 126 mg/dL, range 70-99

    extract_report(_context(db_session, aws, test_settings, document_id, item_id, ai))

    row = _extraction_row(db_session, item_id)
    assert row is not None
    result = row["data"]["results"][0]
    # The flag and conversion are computed by Python, not the model.
    assert result["abnormal_flag"] == "high"  # 126 > 99
    assert result["value_numeric"] == 126.0
    assert result["normalized_unit"] == "mmol/L"
    assert row["prompt_version"] == extraction.PROMPT_VERSION

    log = _logs(db_session, item_id)[0]
    assert log["stage"] == "extracting"
    assert log["outcome"] == "succeeded"


def test_empty_results_is_valid_not_a_failure(db_session, make_document, aws, test_settings):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider(response=structured_response(extraction_payload(results=[])))

    extract_report(_context(db_session, aws, test_settings, document_id, item_id, ai))

    assert _extraction_row(db_session, item_id)["data"]["results"] == []
    assert _logs(db_session, item_id)[0]["outcome"] == "succeeded"


# --- failure handling -------------------------------------------------------


def test_invalid_model_output_is_transient_and_not_persisted(
    db_session, make_document, aws, test_settings
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider(response=structured_response('{"results": [{"value":'))  # broken JSON

    with pytest.raises(TransientStageError):
        extract_report(_context(db_session, aws, test_settings, document_id, item_id, ai))

    assert _extraction_row(db_session, item_id) is None
    log = _logs(db_session, item_id)[0]
    assert log["outcome"] == "validation_failed"
    assert log["error_code"] == "invalid_model_output"


def test_validation_detail_does_not_leak_model_output(
    db_session, make_document, aws, test_settings
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    secret = "PATIENT-JOHN-DOE-SSN-111"
    # test_name missing (required) but a secret smuggled into another field.
    bad = {"results": [{"value": secret, "unit": secret}], "report_date": None}
    ai = FakeAIProvider(response=structured_response(bad))

    with pytest.raises(TransientStageError):
        extract_report(_context(db_session, aws, test_settings, document_id, item_id, ai))

    detail = _logs(db_session, item_id)[0]["error_detail"] or ""
    assert secret not in detail


# --- idempotency ------------------------------------------------------------


def test_rerun_same_attempt_updates_rather_than_duplicates(
    db_session, make_document, aws, test_settings
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ctx = _context(db_session, aws, test_settings, document_id, item_id, FakeAIProvider())

    extract_report(ctx)
    extract_report(ctx)  # redelivery re-runs the same attempt

    assert len(_logs(db_session, item_id)) == 1
    count = db_session.execute(
        text("SELECT count(*) FROM ai_report_extractions WHERE run_item_id = :id"),
        {"id": item_id},
    ).scalar_one()
    assert count == 1
