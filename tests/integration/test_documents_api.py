"""The per-document AI-result endpoints: read a result, and retry a document."""

import json
import uuid
from typing import NamedTuple

import pytest
from sqlalchemy import text

from .conftest import BUCKET

pytestmark = pytest.mark.integration


def _seed_item(
    db_session,
    document_id,
    status,
    section_row_id=None,
    intended_section=None,
    filed_section=None,
    source_key=None,
) -> uuid.UUID:
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items "
            "(run_id, document_id, status, section_row_id, intended_section, "
            "filed_section, source_key) "
            "VALUES (:r, :d, :s, :rep, :sec, :filed, :key) RETURNING id"
        ),
        {
            "r": run_id,
            "d": document_id,
            "s": status,
            "rep": section_row_id,
            "sec": intended_section,
            "filed": filed_section,
            "key": source_key,
        },
    ).scalar_one()
    db_session.flush()
    return item_id


class Filed(NamedTuple):
    document_id: int
    item_id: uuid.UUID
    source_key: str


@pytest.fixture
def filed_failed_item(db_session, aws, seed_user, make_document) -> Filed:
    """A document filed into `reports`, whose later AI stage then failed.

    The state retry now has to serve, reproduced exactly: filing deleted the intake row
    and relocated the object under `reports/`, and the only remaining record of where the
    document lives is the run item's `source_key`.

    The intake row is created and then deleted rather than never created, so the id is one
    the intake sequence really issued — a made-up id would not prove the lookup falls back
    rather than merely missing.
    """
    key = f"reports/{uuid.uuid4().hex}.pdf"
    aws[0].put_object(Bucket=BUCKET, Key=key, Body=b"%PDF-1.4 filed report")

    document_id = make_document(upload=False)
    db_session.execute(text("DELETE FROM unclassified_files WHERE id = :id"), {"id": document_id})
    section_row_id = db_session.execute(
        text(
            "INSERT INTO reports (user_id, created_by, filepath, content) "
            "VALUES (:u, :u, :k, CAST(:c AS JSONB)) RETURNING id"
        ),
        {"u": seed_user, "k": key, "c": json.dumps({"ai": {"state": "classified"}})},
    ).scalar_one()

    item_id = _seed_item(
        db_session,
        document_id,
        "failed",
        section_row_id=section_row_id,
        filed_section="reports",
        source_key=key,
    )
    return Filed(document_id=document_id, item_id=item_id, source_key=key)


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

    response = api.post(f"/v1/documents/reports/{document_id}/ai-result/retry")

    assert response.status_code == 202
    body = response.json()
    assert body["document_id"] == document_id
    assert body["status"] == "queued"  # re-published
    assert uuid.UUID(body["item_id"]) != failed_item  # a fresh attempt


def test_retry_preserves_the_original_intended_section(api, db_session, make_document):
    """A retry must carry the user's original choice forward, or a document rejected for a
    section mismatch would be reprocessed on retry as if it had been uploaded globally --
    the mismatch check defeated on the second attempt, with nothing failing to say so."""
    document_id = make_document()
    old_item = _seed_item(db_session, document_id, "failed", intended_section="vaccinations")

    response = api.post(f"/v1/documents/reports/{document_id}/ai-result/retry")

    assert response.status_code == 202
    new_item_id = uuid.UUID(response.json()["item_id"])
    assert new_item_id != old_item  # a fresh attempt, not the seeded one

    # Assert on the NEW item: the old row keeps its value either way, so checking it would
    # pass even if retry_document dropped the field.
    new_intended = db_session.execute(
        text("SELECT intended_section FROM ai_processing_run_items WHERE id = :id"),
        {"id": new_item_id},
    ).scalar_one()
    assert new_intended == "vaccinations"


def test_retry_a_completed_document_is_rejected(api, db_session, make_document):
    document_id = make_document()
    _seed_item(db_session, document_id, "completed", section_row_id=5)

    response = api.post(f"/v1/documents/reports/{document_id}/ai-result/retry")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "already_completed"


def test_retry_an_in_flight_document_is_rejected(api, db_session, make_document):
    document_id = make_document()
    _seed_item(db_session, document_id, "processing")

    response = api.post(f"/v1/documents/reports/{document_id}/ai-result/retry")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "already_in_progress"


def test_retry_a_never_processed_document_is_404(api, make_document):
    document_id = make_document()
    response = api.post(f"/v1/documents/reports/{document_id}/ai-result/retry")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_ai_result"


def test_retry_a_filed_but_failed_document(api, filed_failed_item):
    """Filing does not end the story: a document filed with classification-only content
    can be reprocessed in place once whatever failed is fixed. Before the run item's
    source_key was consulted this 404'd, because filing had deleted the intake row."""
    response = api.post(f"/v1/documents/reports/{filed_failed_item.document_id}/ai-result/retry")

    assert response.status_code == 202
    assert response.json()["status"] in {"queued", "pending"}


def test_retry_resolves_the_source_from_the_filed_key(api, db_session, filed_failed_item):
    """The intake row is gone; the source is found through the item's source_key -- and
    carried onto the new item, or the worker would have nothing to load the document by."""
    assert filed_failed_item.source_key.startswith("reports/")

    response = api.post(f"/v1/documents/reports/{filed_failed_item.document_id}/ai-result/retry")

    assert response.status_code == 202
    new_item_id = uuid.UUID(response.json()["item_id"])
    assert new_item_id != filed_failed_item.item_id
    new_key = db_session.execute(
        text("SELECT source_key FROM ai_processing_run_items WHERE id = :id"),
        {"id": new_item_id},
    ).scalar_one()
    assert new_key == filed_failed_item.source_key


def test_retry_of_a_document_that_never_existed_is_still_404(api):
    """The fallback must not turn a genuinely unknown id into work."""
    response = api.post("/v1/documents/reports/99999999/ai-result/retry")

    assert response.status_code == 404


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
    # 'medical_condition' is a real section but has no addressable type (it is entered by
    # hand, so there is no AI result to read); FastAPI rejects the path.
    response = api.get("/v1/documents/medical_condition/1/ai-result")

    assert response.status_code == 422


def test_typed_retry_runs_for_the_right_type(api, db_session, make_document):
    document_id = make_document()
    _seed_item(db_session, document_id, "failed")

    response = api.post(f"/v1/documents/reports/{document_id}/ai-result/retry")

    assert response.status_code == 202
    assert response.json()["status"] == "queued"


def test_typed_retry_refuses_the_wrong_type_before_checking_status(api, db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, "failed")
    _seed_classified(db_session, item_id, document_id, "reports")

    response = api.post(f"/v1/documents/vaccinations/{document_id}/ai-result/retry")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "section_mismatch"


def test_typed_retry_allows_a_document_that_never_got_classified(api, db_session, make_document):
    """Deliberately unlike the GET: a document that failed *during* classification has no
    section, and that is exactly when a retry is wanted."""
    document_id = make_document()
    _seed_item(db_session, document_id, "failed")  # no classification row

    read = api.get(f"/v1/documents/reports/{document_id}/ai-result")
    retry = api.post(f"/v1/documents/reports/{document_id}/ai-result/retry")

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


# --- GET status -------------------------------------------------------------
# The untyped route. It exists so Spring need not persist a run_id, so the cases that
# matter are the ones where a *typed* route cannot answer at all.


def test_status_names_the_type_a_result_is_readable_under(api, db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, "completed", section_row_id=77)
    _seed_results(db_session, item_id, document_id)

    response = api.get(f"/v1/documents/{document_id}/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    # The whole point: the caller now knows which URL to build.
    assert body["document_type"] == "reports"
    assert body["section_row_id"] == 77


def test_status_answers_before_classification_where_a_typed_read_cannot(
    api, db_session, make_document
):
    document_id = make_document()
    _seed_item(db_session, document_id, "classifying")

    response = api.get(f"/v1/documents/{document_id}/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "classifying"
    # No classification yet, so no URL to build — but the status is still observable,
    # where GET .../reports/{id}/ai-result would 409 not_classified_yet.
    assert body["document_type"] is None
    assert body["section_row_id"] is None


def test_status_reports_a_rejection_reason(api, db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, "rejected", intended_section="reports")
    _seed_classified(db_session, item_id, document_id, "insurance")
    db_session.execute(
        text(
            "UPDATE ai_processing_run_items SET last_error_code = 'section_mismatch' WHERE id = :i"
        ),
        {"i": item_id},
    )
    db_session.flush()

    body = api.get(f"/v1/documents/{document_id}/status").json()

    assert body["status"] == "rejected"
    assert body["last_error_code"] == "section_mismatch"
    # Classified as insurance despite being uploaded into reports.
    assert body["document_type"] == "insurance"


def test_status_has_no_type_for_a_section_with_no_result_url(api, db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, "rejected")
    _seed_classified(db_session, item_id, document_id, "medical_condition")

    body = api.get(f"/v1/documents/{document_id}/status").json()

    assert body["status"] == "rejected"
    assert body["document_type"] is None


def test_status_carries_nothing_extracted(api, db_session, make_document):
    """The reason this route may be untyped at all."""
    document_id = make_document()
    item_id = _seed_item(db_session, document_id, "completed", section_row_id=5)
    _seed_results(db_session, item_id, document_id)

    body = api.get(f"/v1/documents/{document_id}/status").json()

    for leaked in ("extraction", "insights", "section_extraction", "classification"):
        assert leaked not in body


def test_status_404s_for_a_document_with_no_item(api, make_document):
    response = api.get(f"/v1/documents/{make_document()}/status")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_ai_result"
