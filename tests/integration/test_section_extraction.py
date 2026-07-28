"""The section-extraction stage: persistence, section routing, failure handling, logging.

Runs against the live DB and moto S3 with a fake AI provider, so every branch around the
model call is exercised without a real (paid, non-deterministic) call.
"""

import uuid

import fitz
import pytest
from sqlalchemy import text

from app.integrations.ai.base import AIProviderError
from app.services.section_extraction import extract_section
from app.workers.stagetypes import RejectStageError, StageContext, TransientStageError
from tests.support.ai import FakeAIProvider, structured_response

pytestmark = pytest.mark.integration

#: The stage now reads the document's TEXT, so the fixture body must be a real PDF
#: rather than the placeholder bytes the other stages get away with.
SAMPLE_TEXT = (
    "Star Health Insurance\n"
    "Policy Period 01/10/2019 to 30/09/2020\n"
    "Co-pay 20%\n"
)


def _pdf(body: str = SAMPLE_TEXT) -> bytes:
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), body, fontsize=11)
    data: bytes = doc.tobytes()
    doc.close()
    return data


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


def _seed_classification(db_session, item_id, document_id, section: str) -> None:
    """The stage reads the section from here — it never re-classifies."""
    db_session.execute(
        text(
            "INSERT INTO ai_report_classifications "
            "(run_item_id, document_id, section, title, confidence, prompt_version, "
            "schema_version) VALUES (:i, :d, :s, 'Test Document', 0.95, 'clf-1', 'clf-1')"
        ),
        {"i": item_id, "d": document_id, "s": section},
    )
    db_session.flush()


def _context(db_session, aws, test_settings, document_id, item_id, ai, attempt=1) -> StageContext:
    return StageContext(
        item_id=item_id,
        run_id=uuid.uuid4(),
        document_id=document_id,
        attempt=attempt,
        session=db_session,
        s3=aws[0],
        ai=ai,
        settings=test_settings,
    )


def _extraction_row(db_session, item_id):
    return (
        db_session.execute(
            text("SELECT * FROM ai_section_extractions WHERE run_item_id = :id"),
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


INSURANCE_PAYLOAD = {
    "insurer": "Star Health",
    "policy_name": "Family Health Optima",
    "policy_type": "Health Insurance (Family Floater)",
    "co_pay": "20%",
    "start_date": "1st October 2019",
    "end_date": "30/09/2020",
    "covered_conditions": [{"name": "Maternity", "cap": "50,000"}],
    "exclusions": [{"title": "Cosmetic surgery"}],
}

SCAN_PAYLOAD = {
    "scan_type": "X-Ray",
    "body_part": "Left Knee",
    "scan_date": "31/08/2021 18:11:28",
    "facility": "Apollo Diagnostics",
    "summary": "An X-ray of the left knee. Everything looks normal.",
    "impression": "Normal study.",
    "findings": ["No fracture", "Joint spaces preserved"],
}

VACCINATION_PAYLOAD = {
    "title": "COVID-19 (Covishield) - Dose 2",
    "vaccine_name": "Covishield",
    "dose_info": "Dose 2 of 2",
    "date_given": "23/12/2021",
    "next_due_date": None,
    "facility": "City Clinic",
}


def test_insurance_extraction_persists_iso_dates(db_session, aws, test_settings, make_document):
    document_id = make_document(body=_pdf())
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "insurance")
    ai = FakeAIProvider(response=structured_response(INSURANCE_PAYLOAD))

    extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))

    row = _extraction_row(db_session, item_id)
    assert row is not None
    assert row["section"] == "insurance"
    fields = row["data"]["fields"]
    assert fields["insurer"] == "Star Health"
    # Written as "1st October 2019" on the document; stored normalised.
    assert fields["start_date"] == "2019-10-01"
    assert fields["end_date"] == "2020-09-30"
    assert row["data"]["flags"] == []


def test_scan_extraction_strips_a_study_timestamp(db_session, aws, test_settings, make_document):
    document_id = make_document(body=_pdf())
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "scans_imaging")
    ai = FakeAIProvider(response=structured_response(SCAN_PAYLOAD))

    extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))

    row = _extraction_row(db_session, item_id)
    assert row["section"] == "scans_imaging"
    assert row["data"]["fields"]["scan_date"] == "2021-08-31"
    assert row["data"]["fields"]["findings"] == ["No fracture", "Joint spaces preserved"]


def test_vaccination_extraction_allows_no_next_dose(
    db_session, aws, test_settings, make_document
):
    document_id = make_document(body=_pdf())
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "vaccinations")
    ai = FakeAIProvider(response=structured_response(VACCINATION_PAYLOAD))

    extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))

    fields = _extraction_row(db_session, item_id)["data"]["fields"]
    assert fields["date_given"] == "2021-12-23"
    assert fields["next_due_date"] is None


def test_inverted_policy_period_is_flagged_not_discarded(
    db_session, aws, test_settings, make_document
):
    """A bad date pair is recorded as a data-quality flag; the values still persist."""
    document_id = make_document(body=_pdf())
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "insurance")
    payload = {**INSURANCE_PAYLOAD, "start_date": "29/07/2027", "end_date": "27/07/2026"}
    ai = FakeAIProvider(response=structured_response(payload))

    extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))

    data = _extraction_row(db_session, item_id)["data"]
    assert [f["code"] for f in data["flags"]] == ["dates_out_of_order"]
    assert data["fields"]["start_date"] == "2027-07-29"


def test_rerunning_the_stage_upserts_rather_than_duplicating(
    db_session, aws, test_settings, make_document
):
    """A redelivered SQS message re-runs the stage; it must not append a second row."""
    document_id = make_document(body=_pdf())
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "insurance")
    ai = FakeAIProvider(response=structured_response(INSURANCE_PAYLOAD))
    ctx = _context(db_session, aws, test_settings, document_id, item_id, ai)

    extract_section(ctx)
    ai.set_response(structured_response({**INSURANCE_PAYLOAD, "insurer": "Niva Bupa"}))
    extract_section(ctx)

    rows = db_session.execute(
        text("SELECT count(*) FROM ai_section_extractions WHERE run_item_id = :id"),
        {"id": item_id},
    ).scalar_one()
    assert rows == 1
    assert _extraction_row(db_session, item_id)["data"]["fields"]["insurer"] == "Niva Bupa"
    # Same attempt re-run: one log row, so the cost is not double-counted.
    assert len(_logs(db_session, item_id)) == 1


def test_unhandled_section_is_rejected_not_retried(
    db_session, aws, test_settings, make_document
):
    """Bills classify correctly but have no extractor — terminal, not a retry loop."""
    document_id = make_document(body=_pdf())
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "bills")
    ai = FakeAIProvider(response=structured_response(INSURANCE_PAYLOAD))

    with pytest.raises(RejectStageError) as exc:
        extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))

    assert exc.value.code == "bills"
    assert _extraction_row(db_session, item_id) is None


def test_missing_classification_is_transient(db_session, aws, test_settings, make_document):
    """Classification always runs first; its absence means an interrupted pipeline."""
    document_id = make_document(body=_pdf())
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider(response=structured_response(INSURANCE_PAYLOAD))

    with pytest.raises(TransientStageError):
        extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))


def test_invalid_model_output_is_logged_and_retried(
    db_session, aws, test_settings, make_document
):
    """Model output is never repaired — the failure is recorded and the item retries."""
    document_id = make_document(body=_pdf())
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "insurance")
    # covered_conditions is required by the schema; omitting it must not be patched up.
    ai = FakeAIProvider(response=structured_response({"insurer": "Star Health"}))

    with pytest.raises(TransientStageError):
        extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))

    assert _extraction_row(db_session, item_id) is None
    log = _logs(db_session, item_id)[0]
    assert log["outcome"] == "validation_failed"
    assert log["error_code"] == "invalid_model_output"


def test_provider_error_is_logged_and_retried(db_session, aws, test_settings, make_document):
    document_id = make_document(body=_pdf())
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "insurance")
    ai = FakeAIProvider(error=AIProviderError("upstream timeout"))

    with pytest.raises(TransientStageError):
        extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))

    log = _logs(db_session, item_id)[0]
    assert log["outcome"] == "error"
    assert log["error_code"] == "ai_provider_error"


def test_model_refusal_is_logged_and_retried(db_session, aws, test_settings, make_document):
    document_id = make_document(body=_pdf())
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "scans_imaging")
    ai = FakeAIProvider(response=structured_response(SCAN_PAYLOAD, stop_reason="refusal"))

    with pytest.raises(TransientStageError):
        extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))

    log = _logs(db_session, item_id)[0]
    assert log["outcome"] == "refused"
    assert log["error_code"] == "model_refusal"


def test_the_section_prompt_is_the_one_sent(db_session, aws, test_settings, make_document):
    """Each section must get its own prompt and schema, not a shared default."""
    document_id = make_document(body=_pdf())
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "vaccinations")
    ai = FakeAIProvider(response=structured_response(VACCINATION_PAYLOAD))

    extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))

    call = ai.calls[-1]
    assert "vaccination record" in call["system"]
    assert set(call["json_schema"]["properties"]) == {
        "title",
        "vaccine_name",
        "dose_info",
        "date_given",
        "next_due_date",
        "facility",
    }
