"""The section-extraction stage: persistence, section routing, failure handling, logging.

Runs against the live DB and moto S3 with a fake AI provider, so every branch around the
model call is exercised without a real (paid, non-deterministic) call.
"""

import uuid

import pytest
from sqlalchemy import text

from app.integrations.ai.base import AIProviderError
from app.services.section_extraction import extract_section
from app.services.section_specs import VaccinationFields
from app.workers.stagetypes import RejectStageError, StageContext, TransientStageError
from tests.integration.conftest import document_key
from tests.support.ai import FakeAIProvider, structured_response
from tests.support.pdfs import text_pdf

pytestmark = pytest.mark.integration

#: The stage now reads the document's TEXT, so the fixture body must be a real PDF
#: rather than the placeholder bytes the other stages get away with.
SAMPLE_TEXT = "Star Health Insurance\nPolicy Period 01/10/2019 to 30/09/2020\nCo-pay 20%\n"


def _pdf(body: str = SAMPLE_TEXT) -> bytes:
    """A real single-page PDF with a text layer, authored with reportlab (dev-only)."""
    return text_pdf(body)


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


def test_vaccination_extraction_allows_no_next_dose(db_session, aws, test_settings, make_document):
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


def test_a_scan_with_no_report_completes_and_says_why(
    db_session, aws, test_settings, make_document
):
    """A photographed X-ray with a burned-in header and no radiologist's read.

    The real case from production. It must NOT reject — filing has already happened, so a
    reject stamps the filed row failed and the user is shown a broken document for a file
    we handled correctly. It completes, keeps the factual fields, and explains the empty
    half.

    Unlike the OCR era it DOES reach the model now: with the document going to vision there
    is no text length to pre-check, and a bare image is a fraction of a cent rather than a
    branch. What has to survive is the guard — the model can see the picture, and nothing
    it sees may become a finding.
    """
    document_id = make_document(body=_pdf("20cm"))
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "scans_imaging")
    # What a well-behaved model returns for an image with no report on it: the printed
    # header transcribed, and nothing invented from the picture.
    ai = FakeAIProvider(
        response=structured_response(
            {**SCAN_PAYLOAD, "summary": None, "impression": None, "findings": []}
        )
    )

    extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))

    row = _extraction_row(db_session, item_id)
    assert row is not None
    assert row["section"] == "scans_imaging"
    # The factual transcription survives — that is the half worth having.
    assert row["data"]["fields"]["scan_type"] == "X-Ray"
    assert row["data"]["fields"]["summary"] is None
    codes = {flag["code"] for flag in row["data"]["flags"]}
    # One explanation, not two. Scans say it in their own words; the generic
    # nothing_extracted would only repeat it, and the app gives each flag its own line.
    assert "no_radiologist_read" in codes
    assert "nothing_extracted" not in codes


def test_a_summary_invented_from_the_image_is_still_dropped(
    db_session, aws, test_settings, make_document
):
    """The failure this change made newly reachable, and the backstop that catches it.

    Under OCR a bare X-ray yielded four characters and the model had nothing to work from.
    It can see the image now, so a model that ignores the prompt's ban will write a
    confident all-clear — which is diagnosis, and nothing downstream could contradict it.
    `_drop_unsourced_summary` is the Python half of that line: a summary survives only when
    an impression or a finding it could have been written from survives.
    """
    document_id = make_document(body=_pdf("20cm"))
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "scans_imaging")
    ai = FakeAIProvider(
        response=structured_response(
            {
                **SCAN_PAYLOAD,
                "summary": "The radiologist found no problems with the heart or lungs.",
                "impression": None,
                "findings": [],
            }
        )
    )

    extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))

    row = _extraction_row(db_session, item_id)
    assert row["data"]["fields"]["summary"] is None
    assert "no_radiologist_read" in {flag["code"] for flag in row["data"]["flags"]}


def test_a_section_read_as_entirely_empty_gets_the_generic_explanation(
    db_session, aws, test_settings, make_document
):
    """Insurance has no `no_radiologist_read` equivalent, so it must still say something.

    Without this the policy completes with an empty card and nothing explaining it, and an
    empty card with no reason reads as "the AI failed" rather than "there was nothing here
    to read".
    """
    document_id = make_document(body=_pdf("20cm"))
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "insurance")
    empty = dict.fromkeys(INSURANCE_PAYLOAD)
    ai = FakeAIProvider(
        response=structured_response({**empty, "covered_conditions": [], "exclusions": []})
    )

    extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))

    codes = {flag["code"] for flag in _extraction_row(db_session, item_id)["data"]["flags"]}
    assert "nothing_extracted" in codes


def test_a_document_that_was_read_gets_no_empty_card_flag(
    db_session, aws, test_settings, make_document
):
    """The other direction: one field read is enough to mean the card is not empty."""
    document_id = make_document(body=_pdf("Covishield Dose 2 on 23/12/2021"))
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "vaccinations")
    ai = FakeAIProvider(response=structured_response(VACCINATION_PAYLOAD))

    extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))

    row = _extraction_row(db_session, item_id)
    assert len(ai.calls) == 1
    assert row["data"]["fields"]["vaccine_name"] == "Covishield"
    assert "nothing_extracted" not in {flag["code"] for flag in row["data"]["flags"]}


def test_the_document_itself_is_sent_not_its_text(db_session, aws, test_settings, make_document):
    """The whole change, pinned: this stage sends the FILE, like reports and prescriptions.

    Sending text instead is what put a benefit table's neighbouring column into a sum
    insured, and it is the regression this test exists to catch.
    """
    document_id = make_document(body=_pdf())
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "insurance")
    ai = FakeAIProvider(response=structured_response(INSURANCE_PAYLOAD))

    extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))

    call = ai.calls[-1]
    assert call["document"] is not None, "the stage sent text, not the document"
    assert call["document"].content_type == "application/pdf"
    assert call["document"].data.startswith(b"%PDF")


def test_unhandled_section_is_rejected_not_retried(db_session, aws, test_settings, make_document):
    """A medical condition classifies correctly but has no extractor — terminal, not a
    retry loop."""
    document_id = make_document(body=_pdf())
    item_id = _seed_item(db_session, document_id)
    _seed_classification(db_session, item_id, document_id, "medical_condition")
    ai = FakeAIProvider(response=structured_response(INSURANCE_PAYLOAD))

    with pytest.raises(RejectStageError) as exc:
        extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))

    assert exc.value.code == "medical_condition"
    assert _extraction_row(db_session, item_id) is None


def test_missing_classification_is_transient(db_session, aws, test_settings, make_document):
    """Classification always runs first; its absence means an interrupted pipeline."""
    document_id = make_document(body=_pdf())
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider(response=structured_response(INSURANCE_PAYLOAD))

    with pytest.raises(TransientStageError):
        extract_section(_context(db_session, aws, test_settings, document_id, item_id, ai))


def test_invalid_model_output_is_logged_and_retried(db_session, aws, test_settings, make_document):
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
    # Derived from the model rather than restated: this list used to be a fourth
    # hand-written copy of the vaccination fields, so adding `next_due_interval` broke a
    # test that was checking prompt routing and had no opinion about the field set.
    # Comparing the two also asserts something worth asserting — that the schema the model
    # is sent and the model that validates its answer describe the same shape.
    assert set(call["json_schema"]["properties"]) == set(VaccinationFields.model_fields)
    assert set(call["json_schema"]["required"]) == set(VaccinationFields.model_fields)
