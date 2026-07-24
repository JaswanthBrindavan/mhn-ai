"""The per-document AI-result endpoints: read a result, and retry a document."""

import json
import uuid

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration


def _seed_item(db_session, document_id, status, reports_id=None) -> uuid.UUID:
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status, reports_id) "
            "VALUES (:r, :d, :s, :rep) RETURNING id"
        ),
        {"r": run_id, "d": document_id, "s": status, "rep": reports_id},
    ).scalar_one()
    db_session.flush()
    return item_id


def _seed_results(db_session, item_id, document_id) -> None:
    db_session.execute(
        text(
            "INSERT INTO ai_report_classifications (run_item_id, document_id, section, title, "
            "confidence, prompt_version, schema_version) "
            "VALUES (:i, :d, 'reports', 'CBC', 0.95, 'clf-2', 'clf-2')"
        ),
        {"i": item_id, "d": document_id},
    )
    db_session.execute(
        text(
            "INSERT INTO ai_report_extractions (run_item_id, document_id, data, prompt_version, "
            "schema_version) VALUES (:i, :d, CAST(:data AS JSONB), 'ext-1', 'ext-1')"
        ),
        {
            "i": item_id,
            "d": document_id,
            "data": json.dumps({"results": [{"test_name": "Glucose"}], "report_date": None}),
        },
    )
    db_session.execute(
        text(
            "INSERT INTO ai_report_insights (run_item_id, document_id, data, prompt_version, "
            "schema_version) VALUES (:i, :d, CAST(:data AS JSONB), 'ins-1', 'ins-1')"
        ),
        {
            "i": item_id,
            "d": document_id,
            "data": json.dumps({"insights": [], "summary": None, "disclaimer": "info only"}),
        },
    )
    db_session.flush()


# --- GET ai-result ----------------------------------------------------------


def test_get_ai_result_returns_all_stages(api, db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, "completed", reports_id=77)
    _seed_results(db_session, item_id, document_id)

    response = api.get(f"/v1/documents/{document_id}/ai-result")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["reports_id"] == 77
    assert body["classification"]["section"] == "reports"
    assert body["extraction"]["results"][0]["test_name"] == "Glucose"
    assert body["insights"]["disclaimer"] == "info only"


def test_get_ai_result_for_unprocessed_document_is_404(api, make_document):
    document_id = make_document()  # exists, but never submitted
    response = api.get(f"/v1/documents/{document_id}/ai-result")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_ai_result"


def test_get_ai_result_before_stages_run_has_null_sections(api, db_session, make_document):
    document_id = make_document()
    _seed_item(db_session, document_id, "classifying")

    body = api.get(f"/v1/documents/{document_id}/ai-result").json()

    assert body["status"] == "classifying"
    assert body["classification"] is None
    assert body["extraction"] is None
    assert body["reports_id"] is None


# --- retry ------------------------------------------------------------------


def test_retry_a_failed_document_creates_and_queues_a_new_item(api, db_session, make_document):
    document_id = make_document()
    failed_item = _seed_item(db_session, document_id, "failed")

    response = api.post(f"/v1/documents/{document_id}/ai-result:retry")

    assert response.status_code == 202
    body = response.json()
    assert body["document_id"] == document_id
    assert body["status"] == "queued"  # re-published
    assert uuid.UUID(body["item_id"]) != failed_item  # a fresh attempt


def test_retry_a_completed_document_is_rejected(api, db_session, make_document):
    document_id = make_document()
    _seed_item(db_session, document_id, "completed", reports_id=5)

    response = api.post(f"/v1/documents/{document_id}/ai-result:retry")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "already_completed"


def test_retry_an_in_flight_document_is_rejected(api, db_session, make_document):
    document_id = make_document()
    _seed_item(db_session, document_id, "processing")

    response = api.post(f"/v1/documents/{document_id}/ai-result:retry")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "already_in_progress"


def test_retry_a_never_processed_document_is_404(api, make_document):
    document_id = make_document()
    response = api.post(f"/v1/documents/{document_id}/ai-result:retry")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_ai_result"
