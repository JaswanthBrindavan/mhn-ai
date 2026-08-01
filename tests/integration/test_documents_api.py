"""The per-document AI-result endpoints: read a result, and retry a document."""

import json
import uuid

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration


def _seed_item(db_session, document_id, status, section_row_id=None) -> uuid.UUID:
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status, section_row_id) "
            "VALUES (:r, :d, :s, :rep) RETURNING id"
        ),
        {"r": run_id, "d": document_id, "s": status, "rep": section_row_id},
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
    item_id = _seed_item(db_session, document_id, "completed", section_row_id=77)
    _seed_results(db_session, item_id, document_id)

    response = api.get(f"/v1/documents/reports/{document_id}/ai-result")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["section_row_id"] == 77
    assert body["classification"]["section"] == "reports"
    assert body["extraction"]["results"][0]["test_name"] == "Glucose"
    assert body["insights"]["disclaimer"] == "info only"


# --- retry ------------------------------------------------------------------


def test_retry_a_failed_document_creates_and_queues_a_new_item(api, db_session, make_document):
    document_id = make_document()
    failed_item = _seed_item(db_session, document_id, "failed")

    response = api.post(f"/v1/documents/reports/{document_id}/ai-result:retry")

    assert response.status_code == 202
    body = response.json()
    assert body["document_id"] == document_id
    assert body["status"] == "queued"  # re-published
    assert uuid.UUID(body["item_id"]) != failed_item  # a fresh attempt


def test_retry_a_completed_document_is_rejected(api, db_session, make_document):
    document_id = make_document()
    _seed_item(db_session, document_id, "completed", section_row_id=5)

    response = api.post(f"/v1/documents/reports/{document_id}/ai-result:retry")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "already_completed"


def test_retry_an_in_flight_document_is_rejected(api, db_session, make_document):
    document_id = make_document()
    _seed_item(db_session, document_id, "processing")

    response = api.post(f"/v1/documents/reports/{document_id}/ai-result:retry")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "already_in_progress"


def test_retry_a_never_processed_document_is_404(api, make_document):
    document_id = make_document()
    response = api.post(f"/v1/documents/reports/{document_id}/ai-result:retry")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_ai_result"


# --- the type in the path ---------------------------------------------------
#
# It is the section the document was CLASSIFIED as, and it is the only way to address a
# result: a caller reading an insurance policy can never be handed a lab report's values.


def _seed_classified(db_session, item_id, document_id, section) -> None:
    db_session.execute(
        text(
            "INSERT INTO ai_report_classifications (run_item_id, document_id, section, title, "
            "confidence, prompt_version, schema_version) "
            "VALUES (:i, :d, :s, 'Doc', 0.9, 'clf-2', 'clf-2')"
        ),
        {"i": item_id, "d": document_id, "s": section},
    )
    db_session.flush()


def test_typed_route_returns_the_result_when_the_type_matches(api, db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, "completed", section_row_id=77)
    _seed_results(db_session, item_id, document_id)

    response = api.get(f"/v1/documents/reports/{document_id}/ai-result")

    assert response.status_code == 200
    body = response.json()
    assert body["section_row_id"] == 77
    assert body["classification"]["section"] == "reports"
    assert body["extraction"]["results"][0]["test_name"] == "Glucose"


@pytest.mark.parametrize(
    ("section", "url_type"),
    [
        ("scans_imaging", "scans"),  # the section and the URL word differ on purpose
        ("insurance", "insurance"),
        ("vaccinations", "vaccinations"),
        ("prescriptions", "prescriptions"),
    ],
)
def test_every_typed_route_matches_its_section(api, db_session, make_document, section, url_type):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, "rejected")
    _seed_classified(db_session, item_id, document_id, section)

    response = api.get(f"/v1/documents/{url_type}/{document_id}/ai-result")

    assert response.status_code == 200
    assert response.json()["classification"]["section"] == section


def test_typed_route_refuses_a_document_of_another_type(api, db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, "completed", section_row_id=9)
    _seed_results(db_session, item_id, document_id)  # classified as 'reports'

    response = api.get(f"/v1/documents/insurance/{document_id}/ai-result")

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "section_mismatch"
    # The real section is named, so the caller can go straight to the right URL.
    assert error["details"]["detected_section"] == "reports"


def test_typed_route_refuses_a_document_that_is_not_classified_yet(api, db_session, make_document):
    document_id = make_document()
    _seed_item(db_session, document_id, "classifying")

    response = api.get(f"/v1/documents/reports/{document_id}/ai-result")

    # 409, not 404: the document exists and is being worked on — only its type is unknown.
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_classified_yet"


def test_typed_route_still_404s_for_a_never_processed_document(api, make_document):
    response = api.get(f"/v1/documents/reports/{make_document()}/ai-result")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_ai_result"


def test_an_unknown_document_type_is_rejected_before_any_lookup(api):
    # 'bills' is a real section but has no addressable type; FastAPI rejects the path.
    response = api.get("/v1/documents/bills/1/ai-result")

    assert response.status_code == 422


def test_typed_retry_runs_for_the_right_type(api, db_session, make_document):
    document_id = make_document()
    _seed_item(db_session, document_id, "failed")

    response = api.post(f"/v1/documents/reports/{document_id}/ai-result:retry")

    assert response.status_code == 202
    assert response.json()["status"] == "queued"


def test_typed_retry_refuses_the_wrong_type_before_checking_status(api, db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, "failed")
    _seed_classified(db_session, item_id, document_id, "reports")

    response = api.post(f"/v1/documents/vaccinations/{document_id}/ai-result:retry")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "section_mismatch"


def test_typed_retry_allows_a_document_that_never_got_classified(api, db_session, make_document):
    """Deliberately unlike the GET: a document that failed *during* classification has no
    section, and that is exactly when a retry is wanted."""
    document_id = make_document()
    _seed_item(db_session, document_id, "failed")  # no classification row

    read = api.get(f"/v1/documents/reports/{document_id}/ai-result")
    retry = api.post(f"/v1/documents/reports/{document_id}/ai-result:retry")

    assert read.status_code == 409  # refuses to answer under an unverified type
    assert retry.status_code == 202  # but re-queues the work


def test_a_section_result_is_readable_under_its_own_type(api, db_session, make_document):
    """A non-report section writes ai_section_extractions, not ai_report_extractions.

    Without its own field the whole section payload is invisible through the API: the
    document reads back as classified with a null extraction, and everything the section
    extractor produced is unreachable.
    """
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, "completed")
    _seed_classified(db_session, item_id, document_id, "insurance")
    db_session.execute(
        text(
            "INSERT INTO ai_section_extractions (run_item_id, document_id, section, data, "
            "prompt_version, schema_version) "
            "VALUES (:i, :d, 'insurance', CAST(:data AS JSONB), 'sec-1', 'sec-1')"
        ),
        {
            "i": item_id,
            "d": document_id,
            "data": json.dumps(
                {
                    "section": "insurance",
                    "fields": {"insurer": "Star Health", "co_pay": "20%"},
                    "flags": [],
                }
            ),
        },
    )
    db_session.flush()

    body = api.get(f"/v1/documents/insurance/{document_id}/ai-result").json()

    assert body["section_extraction"]["fields"]["insurer"] == "Star Health"
    assert body["section_extraction"]["flags"] == []
    # The report-shaped fields stay null: the two are mutually exclusive.
    assert body["extraction"] is None
    assert body["insights"] is None


def test_a_report_result_carries_no_section_extraction(api, db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, "completed", section_row_id=3)
    _seed_results(db_session, item_id, document_id)

    body = api.get(f"/v1/documents/reports/{document_id}/ai-result").json()

    assert body["extraction"] is not None
    assert body["section_extraction"] is None
