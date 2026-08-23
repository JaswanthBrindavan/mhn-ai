"""The classification stage: persistence, reject rules, failure handling, cost logging.

Runs against the live DB and moto S3 with a fake AI provider, so every branch around
the model call is exercised without a real (paid, non-deterministic) call.
"""

import uuid
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy import text

from app.services import classification
from app.services.classification import classify_report
from app.workers.stagetypes import RejectStageError, StageContext, TransientStageError
from tests.integration.conftest import document_key
from tests.support.ai import FakeAIProvider, classification_payload, structured_response

pytestmark = pytest.mark.integration


def _seed_item(db_session, document_id: int) -> uuid.UUID:
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status) "
            "VALUES (:r, :rep, 'processing') RETURNING id"
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


def _classification_row(db_session, item_id):
    return (
        db_session.execute(
            text("SELECT * FROM ai_report_classifications WHERE run_item_id = :id"),
            {"id": item_id},
        )
        .mappings()
        .one_or_none()
    )


def _logs(db_session, item_id):
    return (
        db_session.execute(
            text(
                "SELECT * FROM ai_process_logs WHERE run_item_id = :id ORDER BY attempt, created_at"
            ),
            {"id": item_id},
        )
        .mappings()
        .all()
    )


# --- happy path -------------------------------------------------------------


def test_processable_report_persists_classification_and_log(
    db_session, make_document, aws, test_settings
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider()  # default: a report

    classify_report(_context(db_session, aws, test_settings, document_id, item_id, ai))

    row = _classification_row(db_session, item_id)
    assert row is not None
    assert row["section"] == "reports"
    assert row["title"] == "Complete Blood Count"
    assert row["prompt_version"] == classification.PROMPT_VERSION

    logs = _logs(db_session, item_id)
    assert len(logs) == 1
    assert logs[0]["outcome"] == "succeeded"
    assert logs[0]["provider"] == "anthropic"
    # 1200 input @ $5/1M + 40 output @ $25/1M = 0.007
    assert logs[0]["estimated_cost_usd"] == Decimal("0.007000")
    assert logs[0]["input_tokens"] == 1200


def test_document_media_type_is_passed_to_the_model(db_session, make_document, aws, test_settings):
    document_id = make_document(content_type="image/png", suffix=".png", body=b"\x89PNG fake")
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider()

    classify_report(_context(db_session, aws, test_settings, document_id, item_id, ai))

    assert ai.last_document.content_type == "image/png"
    assert ai.last_document.data == b"\x89PNG fake"


# --- rejection --------------------------------------------------------------


@pytest.mark.parametrize("section", ["scans_imaging", "unknown", "bills"])
def test_a_non_report_section_is_recorded_and_not_rejected_here(
    db_session, make_document, aws, test_settings, section
):
    """This stage classifies; it does not route.

    Whether a section has a pipeline is the processor's decision (SECTION_PIPELINES), so
    classifying a scan is a *success* here even though the document may go no further.
    Rejecting in this stage would also report a correct classification as a stage failure.
    """
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider(
        response=structured_response(classification_payload(section=section, title="A Document"))
    )

    classify_report(_context(db_session, aws, test_settings, document_id, item_id, ai))

    # Recorded for audit and for the processor to route on, and logged as a success.
    assert _classification_row(db_session, item_id)["section"] == section
    assert _logs(db_session, item_id)[0]["outcome"] == "succeeded"


def test_missing_s3_object_is_rejected(db_session, make_document, aws, test_settings):
    document_id = make_document(upload=False)  # row exists, object does not
    item_id = _seed_item(db_session, document_id)

    with pytest.raises(RejectStageError) as excinfo:
        classify_report(
            _context(db_session, aws, test_settings, document_id, item_id, FakeAIProvider())
        )

    assert excinfo.value.code == "source_object_missing"


# --- failure handling -------------------------------------------------------


def test_invalid_model_output_is_a_transient_failure(db_session, make_document, aws, test_settings):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider(response=structured_response('{"is_report": true, "title":'))  # broken JSON

    with pytest.raises(TransientStageError):
        classify_report(_context(db_session, aws, test_settings, document_id, item_id, ai))

    # Never repaired, never persisted as a result; the failure is logged.
    assert _classification_row(db_session, item_id) is None
    log = _logs(db_session, item_id)[0]
    assert log["outcome"] == "validation_failed"
    assert log["error_code"] == "invalid_model_output"


def test_validation_error_detail_does_not_leak_model_output(
    db_session, make_document, aws, test_settings
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    # Invalid section value plus a made-up patient string that must NOT appear in the log.
    secret = "PATIENT-JANE-DOE-SSN-000"
    ai = FakeAIProvider(
        response=structured_response(classification_payload(section=secret, title=secret))
    )

    with pytest.raises(TransientStageError):
        classify_report(_context(db_session, aws, test_settings, document_id, item_id, ai))

    detail = _logs(db_session, item_id)[0]["error_detail"] or ""
    assert secret not in detail
    assert "section" in detail  # field location is fine to record


def test_refusal_is_a_transient_failure(db_session, make_document, aws, test_settings):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider(response=structured_response("", stop_reason="refusal"))

    with pytest.raises(TransientStageError):
        classify_report(_context(db_session, aws, test_settings, document_id, item_id, ai))

    assert _logs(db_session, item_id)[0]["outcome"] == "refused"


def test_provider_error_is_transient_and_logged(db_session, make_document, aws, test_settings):
    from app.integrations.ai.base import AIProviderError

    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider(error=AIProviderError("Timeout"))

    with pytest.raises(TransientStageError):
        classify_report(_context(db_session, aws, test_settings, document_id, item_id, ai))

    log = _logs(db_session, item_id)[0]
    assert log["outcome"] == "error"
    assert log["input_tokens"] == 0  # no successful call, no tokens


# --- idempotency ------------------------------------------------------------


def test_rerun_same_attempt_updates_rather_than_duplicates(
    db_session, make_document, aws, test_settings
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider()
    ctx = _context(db_session, aws, test_settings, document_id, item_id, ai, attempt=1)

    classify_report(ctx)
    classify_report(ctx)  # same attempt re-runs (e.g. redelivery)

    # One classification, one log row — the attempt's cost is not double-counted.
    assert len(_logs(db_session, item_id)) == 1
    count = db_session.execute(
        text("SELECT count(*) FROM ai_report_classifications WHERE run_item_id = :id"),
        {"id": item_id},
    ).scalar_one()
    assert count == 1


def test_the_chosen_date_and_its_label_are_persisted(db_session, make_document, aws, test_settings):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider(
        response=structured_response(
            classification_payload(
                dates=[
                    {"label": "Reported On", "value": "15/03/2026"},
                    {"label": "Sample Collected", "value": "12/03/2026"},
                ]
            )
        )
    )
    ctx = _context(db_session, aws, test_settings, document_id, item_id, ai)

    classify_report(ctx)

    row = db_session.execute(
        text(
            "SELECT document_date, document_date_label FROM ai_report_classifications "
            "WHERE run_item_id = :id"
        ),
        {"id": item_id},
    ).one()
    assert row.document_date == date(2026, 3, 12)
    assert row.document_date_label == "Sample Collected"


def test_a_document_printing_no_date_stores_null_rather_than_a_guess(
    db_session, make_document, aws, test_settings
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider(response=structured_response(classification_payload(dates=[])))
    ctx = _context(db_session, aws, test_settings, document_id, item_id, ai)

    classify_report(ctx)

    row = db_session.execute(
        text(
            "SELECT document_date, document_date_label FROM ai_report_classifications "
            "WHERE run_item_id = :id"
        ),
        {"id": item_id},
    ).one()
    assert row.document_date is None
    assert row.document_date_label is None


def test_a_rerun_replaces_the_stored_date_rather_than_keeping_the_first(
    db_session, make_document, aws, test_settings
):
    # The upsert path, which is easy to get wrong in one direction only: leaving these
    # columns out of on_conflict_do_update's set_ leaves the first pass's date sitting
    # under the second pass's reading, and nothing anywhere says so.
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)

    first = FakeAIProvider(
        response=structured_response(
            classification_payload(dates=[{"label": "Sample Collected", "value": "12/03/2026"}])
        )
    )
    classify_report(_context(db_session, aws, test_settings, document_id, item_id, first))

    second = FakeAIProvider(
        response=structured_response(
            classification_payload(dates=[{"label": "Sample Collected", "value": "05/01/2026"}])
        )
    )
    classify_report(_context(db_session, aws, test_settings, document_id, item_id, second))

    row = db_session.execute(
        text("SELECT document_date FROM ai_report_classifications WHERE run_item_id = :id"),
        {"id": item_id},
    ).one()
    assert row.document_date == date(2026, 1, 5)


def _finish(db_session, item_id) -> None:
    """Close an item so a second one for the same document may exist.

    The partial unique index allows one in-flight item per document, which is exactly the
    state a resume starts from: the first pass is terminal before the second is created.
    """
    db_session.execute(
        text("UPDATE ai_processing_run_items SET status = 'completed' WHERE id = :i"),
        {"i": item_id},
    )
    db_session.flush()


def test_adopt_prior_copies_the_whole_classification_onto_a_new_item(
    db_session, make_document, aws, test_settings
):
    """A resumed or retried document takes the classification it already has.

    Tested here rather than only through the worker, because two of these fields are also
    restored downstream by identity._carry_settled -- so a worker-level test passes even
    if this function drops them, and would go on passing until someone switched name
    matching off.
    """
    document_id = make_document()
    first_item = _seed_item(db_session, document_id)
    ai = FakeAIProvider(
        response=structured_response(
            classification_payload(dates=[{"label": "Sample Collected", "value": "12/03/2026"}])
        )
    )
    classify_report(_context(db_session, aws, test_settings, document_id, first_item, ai))
    db_session.execute(
        text(
            "UPDATE ai_report_classifications SET patient_name = 'PRIYA MENON', "
            "name_match = 'mismatch', identity_confirmed_at = now() WHERE run_item_id = :i"
        ),
        {"i": first_item},
    )
    _finish(db_session, first_item)

    second_item = _seed_item(db_session, document_id)
    assert (
        classification.adopt_prior(db_session, item_id=second_item, document_id=document_id) is True
    )

    columns = (
        "section, title, confidence, reasoning, patient_name, name_match, "
        "identity_confirmed_at, document_date, document_date_label, prompt_version, "
        "schema_version"
    )
    rows = [
        db_session.execute(
            text(f"SELECT {columns} FROM ai_report_classifications WHERE run_item_id = :i"),
            {"i": item},
        ).one()
        for item in (second_item, first_item)
    ]
    assert rows[0] == rows[1]
    # Spelt out so the equality above cannot pass on two rows of nulls.
    assert rows[0].document_date == date(2026, 3, 12)
    assert rows[0].identity_confirmed_at is not None


def test_adopting_records_no_process_log_and_no_fresh_version(
    db_session, make_document, aws, test_settings
):
    # No model was called, so there is nothing to bill and nothing to stamp. Writing the
    # current PROMPT_VERSION here would claim a reading under a version that never ran --
    # the same lie a skipped insights stage avoids by logging "skipped" instead of the
    # configured model.
    document_id = make_document()
    first_item = _seed_item(db_session, document_id)
    classify_report(
        _context(db_session, aws, test_settings, document_id, first_item, FakeAIProvider())
    )
    db_session.execute(
        text(
            "UPDATE ai_report_classifications SET prompt_version = 'clf-ancient', "
            "schema_version = 'clf-0' WHERE run_item_id = :i"
        ),
        {"i": first_item},
    )
    _finish(db_session, first_item)

    second_item = _seed_item(db_session, document_id)
    classification.adopt_prior(db_session, item_id=second_item, document_id=document_id)

    versions = db_session.execute(
        text(
            "SELECT prompt_version, schema_version FROM ai_report_classifications "
            "WHERE run_item_id = :i"
        ),
        {"i": second_item},
    ).one()
    assert versions.prompt_version == "clf-ancient"
    assert versions.schema_version == "clf-0"
    assert _logs(db_session, second_item) == []


def test_adopt_prior_finds_nothing_for_a_document_never_classified(db_session, make_document):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)

    assert classification.adopt_prior(db_session, item_id=item_id, document_id=document_id) is False


def test_a_new_attempt_adds_a_separate_log_row(db_session, make_document, aws, test_settings):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider()

    classify_report(_context(db_session, aws, test_settings, document_id, item_id, ai, attempt=1))
    classify_report(_context(db_session, aws, test_settings, document_id, item_id, ai, attempt=2))

    logs = _logs(db_session, item_id)
    assert len(logs) == 2  # each real attempt is a separately billed call
    assert {log["attempt"] for log in logs} == {1, 2}
