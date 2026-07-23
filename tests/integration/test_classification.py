"""The classification stage: persistence, reject rules, failure handling, cost logging.

Runs against the live DB and moto S3 with a fake AI provider, so every branch around
the model call is exercised without a real (paid, non-deterministic) call.
"""

import uuid
from decimal import Decimal

import pytest
from sqlalchemy import text

from app.services import classification
from app.services.classification import classify_report
from app.workers.stagetypes import RejectStageError, StageContext, TransientStageError
from tests.support.ai import FakeAIProvider, lab_report_payload, structured_response

pytestmark = pytest.mark.integration


def _seed_item(db_session, report_id: int) -> uuid.UUID:
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, report_id, status) "
            "VALUES (:r, :rep, 'processing') RETURNING id"
        ),
        {"r": run_id, "rep": report_id},
    ).scalar_one()
    db_session.flush()
    return item_id


def _context(db_session, aws, test_settings, report_id, item_id, ai, attempt=1) -> StageContext:
    return StageContext(
        item_id=item_id,
        run_id=uuid.uuid4(),
        report_id=report_id,
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
    db_session, make_report, aws, test_settings
):
    report_id = make_report()
    item_id = _seed_item(db_session, report_id)
    ai = FakeAIProvider()  # default: a lab_report

    classify_report(_context(db_session, aws, test_settings, report_id, item_id, ai))

    row = _classification_row(db_session, item_id)
    assert row is not None
    assert row["is_report"] is True
    assert row["document_type"] == "lab_report"
    assert row["title"] == "Complete Blood Count"
    assert row["prompt_version"] == classification.PROMPT_VERSION

    logs = _logs(db_session, item_id)
    assert len(logs) == 1
    assert logs[0]["outcome"] == "succeeded"
    assert logs[0]["provider"] == "anthropic"
    # 1200 input @ $5/1M + 40 output @ $25/1M = 0.007
    assert logs[0]["estimated_cost_usd"] == Decimal("0.007000")
    assert logs[0]["input_tokens"] == 1200


def test_document_media_type_is_passed_to_the_model(db_session, make_report, aws, test_settings):
    report_id = make_report(content_type="image/png", suffix=".png", body=b"\x89PNG fake")
    item_id = _seed_item(db_session, report_id)
    ai = FakeAIProvider()

    classify_report(_context(db_session, aws, test_settings, report_id, item_id, ai))

    assert ai.last_document.content_type == "image/png"
    assert ai.last_document.data == b"\x89PNG fake"


# --- rejection --------------------------------------------------------------


def test_non_report_is_rejected_but_still_recorded(db_session, make_report, aws, test_settings):
    report_id = make_report()
    item_id = _seed_item(db_session, report_id)
    ai = FakeAIProvider(
        response=structured_response(
            lab_report_payload(is_report=False, document_type="non_medical", title="Flyer")
        )
    )

    with pytest.raises(RejectStageError) as excinfo:
        classify_report(_context(db_session, aws, test_settings, report_id, item_id, ai))

    assert excinfo.value.code == "not_a_report"
    # The classification is still recorded (audit), and the log marks it rejected.
    assert _classification_row(db_session, item_id)["is_report"] is False
    assert _logs(db_session, item_id)[0]["outcome"] == "rejected"


def test_wrong_document_type_is_rejected(db_session, make_report, aws, test_settings):
    report_id = make_report()
    item_id = _seed_item(db_session, report_id)
    # A real medical doc, but a prescription — out of this sprint's scope.
    ai = FakeAIProvider(
        response=structured_response(
            lab_report_payload(is_report=True, document_type="prescription")
        )
    )

    with pytest.raises(RejectStageError) as excinfo:
        classify_report(_context(db_session, aws, test_settings, report_id, item_id, ai))

    assert excinfo.value.code == "wrong_document_type"
    assert _logs(db_session, item_id)[0]["error_code"] == "wrong_document_type"


def test_missing_s3_object_is_rejected(db_session, make_report, aws, test_settings):
    report_id = make_report(upload=False)  # row exists, object does not
    item_id = _seed_item(db_session, report_id)

    with pytest.raises(RejectStageError) as excinfo:
        classify_report(
            _context(db_session, aws, test_settings, report_id, item_id, FakeAIProvider())
        )

    assert excinfo.value.code == "source_object_missing"


# --- failure handling -------------------------------------------------------


def test_invalid_model_output_is_a_transient_failure(db_session, make_report, aws, test_settings):
    report_id = make_report()
    item_id = _seed_item(db_session, report_id)
    ai = FakeAIProvider(response=structured_response('{"is_report": true, "title":'))  # broken JSON

    with pytest.raises(TransientStageError):
        classify_report(_context(db_session, aws, test_settings, report_id, item_id, ai))

    # Never repaired, never persisted as a result; the failure is logged.
    assert _classification_row(db_session, item_id) is None
    log = _logs(db_session, item_id)[0]
    assert log["outcome"] == "validation_failed"
    assert log["error_code"] == "invalid_model_output"


def test_validation_error_detail_does_not_leak_model_output(
    db_session, make_report, aws, test_settings
):
    report_id = make_report()
    item_id = _seed_item(db_session, report_id)
    # Wrong enum value plus a made-up patient string that must NOT appear in the log.
    secret = "PATIENT-JANE-DOE-SSN-000"
    ai = FakeAIProvider(
        response=structured_response(lab_report_payload(document_type=secret, title=secret))
    )

    with pytest.raises(TransientStageError):
        classify_report(_context(db_session, aws, test_settings, report_id, item_id, ai))

    detail = _logs(db_session, item_id)[0]["error_detail"] or ""
    assert secret not in detail
    assert "document_type" in detail  # field location is fine to record


def test_refusal_is_a_transient_failure(db_session, make_report, aws, test_settings):
    report_id = make_report()
    item_id = _seed_item(db_session, report_id)
    ai = FakeAIProvider(response=structured_response("", stop_reason="refusal"))

    with pytest.raises(TransientStageError):
        classify_report(_context(db_session, aws, test_settings, report_id, item_id, ai))

    assert _logs(db_session, item_id)[0]["outcome"] == "refused"


def test_provider_error_is_transient_and_logged(db_session, make_report, aws, test_settings):
    from app.integrations.ai.base import AIProviderError

    report_id = make_report()
    item_id = _seed_item(db_session, report_id)
    ai = FakeAIProvider(error=AIProviderError("Timeout"))

    with pytest.raises(TransientStageError):
        classify_report(_context(db_session, aws, test_settings, report_id, item_id, ai))

    log = _logs(db_session, item_id)[0]
    assert log["outcome"] == "error"
    assert log["input_tokens"] == 0  # no successful call, no tokens


# --- idempotency ------------------------------------------------------------


def test_rerun_same_attempt_updates_rather_than_duplicates(
    db_session, make_report, aws, test_settings
):
    report_id = make_report()
    item_id = _seed_item(db_session, report_id)
    ai = FakeAIProvider()
    ctx = _context(db_session, aws, test_settings, report_id, item_id, ai, attempt=1)

    classify_report(ctx)
    classify_report(ctx)  # same attempt re-runs (e.g. redelivery)

    # One classification, one log row — the attempt's cost is not double-counted.
    assert len(_logs(db_session, item_id)) == 1
    count = db_session.execute(
        text("SELECT count(*) FROM ai_report_classifications WHERE run_item_id = :id"),
        {"id": item_id},
    ).scalar_one()
    assert count == 1


def test_a_new_attempt_adds_a_separate_log_row(db_session, make_report, aws, test_settings):
    report_id = make_report()
    item_id = _seed_item(db_session, report_id)
    ai = FakeAIProvider()

    classify_report(_context(db_session, aws, test_settings, report_id, item_id, ai, attempt=1))
    classify_report(_context(db_session, aws, test_settings, report_id, item_id, ai, attempt=2))

    logs = _logs(db_session, item_id)
    assert len(logs) == 2  # each real attempt is a separately billed call
    assert {log["attempt"] for log in logs} == {1, 2}
