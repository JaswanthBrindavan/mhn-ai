"""End-to-end message processing: claim -> stages -> terminal -> ack.

Uses moto SQS and the live database. Each processed message runs in its own session
bound to the test's transaction, so everything rolls back and nothing leaks into the
Spring-shared database.
"""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.integrations.sqs import publish_processing_item, receive_messages
from app.models.enums import RunItemStatus
from app.services.classification import DocumentSection
from app.services.section_specs import SECTION_SPECS
from app.workers.processor import Outcome, process_message
from tests.support.ai import FakeAIProvider, classification_payload, structured_response
from tests.support.pdfs import text_pdf

pytestmark = pytest.mark.integration

# A processable lab-report classification, so the real classify stage lets the pipeline
# complete. Tests that monkeypatch CLASSIFY_STAGE replace classify, so `ai` is unused there.
_FAKE_AI = FakeAIProvider()


@pytest.fixture
def session_factory(db_connection):
    """Fresh sessions on the test connection, so worker commits stay inside the roll-back."""

    def _make() -> Session:
        return Session(bind=db_connection, join_transaction_mode="create_savepoint")

    return _make


def _seed_item(
    db_session, make_document, status: str = "queued"
) -> tuple[uuid.UUID, uuid.UUID, int]:
    document_id = make_document()
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status) "
            "VALUES (:r, :rep, :s) RETURNING id"
        ),
        {"r": run_id, "rep": document_id, "s": status},
    ).scalar_one()
    db_session.flush()
    return item_id, run_id, document_id


def _status(db_session, item_id) -> str:
    return db_session.execute(
        text("SELECT status FROM ai_processing_run_items WHERE id = :id"), {"id": item_id}
    ).scalar_one()


def _receive_one(sqs, queue_url):
    msgs = receive_messages(sqs, queue_url, max_messages=1, wait_seconds=0, visibility_timeout=30)
    assert len(msgs) == 1
    return msgs[0]


def _queue_depth(sqs, queue_url) -> int:
    attrs = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"],
    )["Attributes"]
    return int(attrs["ApproximateNumberOfMessages"]) + int(
        attrs["ApproximateNumberOfMessagesNotVisible"]
    )


def _process(sqs, queue_url, session_factory, test_settings, aws, *, ai=None):
    s3 = aws[0]
    message = _receive_one(sqs, queue_url)
    return process_message(
        message,
        session_factory=session_factory,
        s3=s3,
        sqs=sqs,
        ai=ai or _FAKE_AI,
        settings=test_settings,
    )


# --- happy path -------------------------------------------------------------


def test_full_pipeline_completes_and_acks(
    db_session, make_document, session_factory, test_settings, aws
):
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.COMPLETED
    assert _status(db_session, item_id) == RunItemStatus.COMPLETED.value
    # The message was deleted, not left to redeliver.
    assert _queue_depth(sqs, queue_url) == 0


def test_completed_at_and_started_at_are_set(
    db_session, make_document, session_factory, test_settings, aws
):
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    _process(sqs, queue_url, session_factory, test_settings, aws)

    row = db_session.execute(
        text(
            "SELECT started_at, completed_at, attempt_count "
            "FROM ai_processing_run_items WHERE id=:id"
        ),
        {"id": item_id},
    ).one()
    assert row.started_at is not None
    assert row.completed_at is not None
    assert row.attempt_count == 1


# --- idempotency / at-least-once --------------------------------------------


def test_duplicate_delivery_of_completed_item_is_a_noop(
    db_session, make_document, session_factory, test_settings, aws
):
    """A redelivered message for an already-completed item must not reprocess it."""
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)
    _process(sqs, queue_url, session_factory, test_settings, aws)  # completes it

    # Same message delivered again (e.g. an earlier delete that never landed).
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)
    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.SKIPPED_TERMINAL
    assert _status(db_session, item_id) == RunItemStatus.COMPLETED.value


def test_message_for_deleted_item_is_acked_as_not_found(
    db_session, make_document, session_factory, test_settings, aws
):
    _, sqs, queue_url, _ = aws
    ghost = uuid.uuid4()
    publish_processing_item(sqs, queue_url, item_id=ghost, run_id=uuid.uuid4(), document_id=1)

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.NOT_FOUND
    assert _queue_depth(sqs, queue_url) == 0  # stale message dropped


# --- cancellation -----------------------------------------------------------


def test_cancelled_before_processing_is_skipped(
    db_session, make_document, session_factory, test_settings, aws
):
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document, status="cancelled")
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.SKIPPED_TERMINAL
    assert _status(db_session, item_id) == RunItemStatus.CANCELLED.value
    assert _queue_depth(sqs, queue_url) == 0


# --- cancellation mid-pipeline ----------------------------------------------


def test_cancellation_during_a_stage_stops_the_pipeline(
    db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    """A cancel that lands while a stage runs must stop the worker cleanly."""
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    from app.models.enums import RunItemStatus as S

    def _cancelling_stage(ctx) -> None:
        # Simulate the DELETE endpoint cancelling this item mid-flight.
        ctx.session.execute(
            text("UPDATE ai_processing_run_items SET status='cancelled' WHERE id=:id"),
            {"id": ctx.item_id},
        )
        ctx.session.commit()

    monkeypatch.setattr("app.workers.processor.CLASSIFY_STAGE", (S.CLASSIFYING, _cancelling_stage))

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.CANCELLED
    assert _status(db_session, item_id) == S.CANCELLED.value
    # Not completed, and the message is dropped (cancelled is terminal).
    assert _queue_depth(sqs, queue_url) == 0


# --- stage failure modes ----------------------------------------------------


def test_transient_stage_failure_leaves_message_for_redelivery(
    db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    from app.models.enums import RunItemStatus as S
    from app.workers.stages import TransientStageError

    def _flaky(_ctx) -> None:
        raise TransientStageError("provider 503")

    monkeypatch.setattr("app.workers.processor.CLASSIFY_STAGE", (S.CLASSIFYING, _flaky))

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.RETRY
    # Not terminal; will be retried on redelivery.
    assert _status(db_session, item_id) not in {"completed", "failed", "rejected", "cancelled"}
    # The message was NOT deleted (still owned, invisible during its timeout).
    assert _queue_depth(sqs, queue_url) == 1


def test_reject_stage_marks_rejected_and_acks(
    db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    from app.models.enums import RunItemStatus as S
    from app.workers.stages import RejectStageError

    def _reject(_ctx) -> None:
        raise RejectStageError("not_a_report", "Document is not a lab report")

    monkeypatch.setattr("app.workers.processor.CLASSIFY_STAGE", (S.CLASSIFYING, _reject))

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.REJECTED
    assert _status(db_session, item_id) == S.REJECTED.value
    code = db_session.execute(
        text("SELECT last_error_code FROM ai_processing_run_items WHERE id=:id"), {"id": item_id}
    ).scalar_one()
    assert code == "not_a_report"
    assert _queue_depth(sqs, queue_url) == 0  # terminal → dropped


# --- attempt cap ------------------------------------------------------------


def test_message_giving_up_after_max_attempts_fails_and_acks(
    db_session, make_document, session_factory, test_settings, aws
):
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    # Push attempt_count to the cap so the next claim gives up.
    db_session.execute(
        text(
            "UPDATE ai_processing_run_items SET attempt_count=:a, status='processing' WHERE id=:id"
        ),
        {"a": test_settings.max_attempts, "id": item_id},
    )
    db_session.flush()
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.GAVE_UP
    assert _status(db_session, item_id) == RunItemStatus.FAILED.value
    assert _queue_depth(sqs, queue_url) == 0


# --- routing: the pipeline's shape depends on the detected section ----------


class _ClassifiesAs(FakeAIProvider):
    """Forces the classification result, leaving every other stage to the schema dispatch.

    Setting a single fixed `response` would not do: one message runs classification *and*
    whatever stage follows it, and they need different payloads.
    """

    def __init__(self, section: str) -> None:
        super().__init__()
        self.section = section

    def analyze_document(self, **kwargs):
        default = super().analyze_document(**kwargs)  # records the call
        if "section" in kwargs["json_schema"].get("properties", {}):
            return structured_response(classification_payload(section=self.section))
        return default


def _section_row(db_session, item_id):
    return (
        db_session.execute(
            text("SELECT section, data FROM ai_section_extractions WHERE run_item_id = :id"),
            {"id": item_id},
        )
        .mappings()
        .one_or_none()
    )


def _source_exists(db_session, document_id) -> bool:
    return (
        db_session.execute(
            text("SELECT count(*) FROM unclassified_files WHERE id = :id"), {"id": document_id}
        ).scalar_one()
        == 1
    )


@pytest.mark.parametrize("section", ["insurance", "scans_imaging", "vaccinations"])
def test_a_section_document_is_extracted_then_stops(
    db_session, make_document, session_factory, test_settings, aws, section
):
    """A non-report section runs extract_section and completes — without being moved.

    The document stays in unclassified_files: filing it is a separate decision (see
    docs/document-filing-design.md), and no reports row must appear for a scan.
    """
    _, sqs, queue_url, _ = aws
    # A real PDF: this path reads the document's text rather than sending the file.
    document_id = make_document(body=text_pdf("Policy Period 01/10/2019 to 30/09/2020"))
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status) "
            "VALUES (:r, :d, 'queued') RETURNING id"
        ),
        {"r": run_id, "d": document_id},
    ).scalar_one()
    db_session.flush()
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(
        sqs, queue_url, session_factory, test_settings, aws, ai=_ClassifiesAs(section)
    )

    assert outcome is Outcome.COMPLETED
    assert _status(db_session, item_id) == RunItemStatus.COMPLETED.value
    row = _section_row(db_session, item_id)
    assert row is not None and row["section"] == section
    # The stored fields are that section's own, proving the right SectionSpec drove the
    # call rather than some other section's schema.
    expected_fields = set(SECTION_SPECS[DocumentSection(section)].json_schema["properties"])
    assert set(row["data"]["fields"]) == expected_fields
    # Not moved, and no reports row invented for a non-report.
    assert _source_exists(db_session, document_id)
    reports_id = db_session.execute(
        text("SELECT reports_id FROM ai_processing_run_items WHERE id = :id"), {"id": item_id}
    ).scalar_one()
    assert reports_id is None
    assert _queue_depth(sqs, queue_url) == 0


@pytest.mark.parametrize("section", ["bills", "medical_condition", "prescriptions", "unknown"])
def test_a_section_with_no_pipeline_is_rejected_by_the_router(
    db_session, make_document, session_factory, test_settings, aws, section
):
    """Routing, not failure: no extractor exists, so the document stays where it is."""
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(
        sqs, queue_url, session_factory, test_settings, aws, ai=_ClassifiesAs(section)
    )

    assert outcome is Outcome.REJECTED
    assert _status(db_session, item_id) == RunItemStatus.REJECTED.value
    # The detected section is the reason, so the caller can route on it.
    code = db_session.execute(
        text("SELECT last_error_code FROM ai_processing_run_items WHERE id = :id"), {"id": item_id}
    ).scalar_one()
    assert code == section
    assert _source_exists(db_session, document_id)
    # The classification is still recorded even though nothing processed it.
    assert (
        db_session.execute(
            text("SELECT section FROM ai_report_classifications WHERE run_item_id = :id"),
            {"id": item_id},
        ).scalar_one()
        == section
    )


def test_a_report_still_moves_and_a_section_never_does(
    db_session, make_document, session_factory, test_settings, aws
):
    """The two paths diverge at the end: only a report is moved into `reports`."""
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.COMPLETED
    reports_id = db_session.execute(
        text("SELECT reports_id FROM ai_processing_run_items WHERE id = :id"), {"id": item_id}
    ).scalar_one()
    assert reports_id is not None
    # Moved: the intake row is gone, unlike every section document.
    assert not _source_exists(db_session, document_id)
