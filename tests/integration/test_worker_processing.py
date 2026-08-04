"""End-to-end message processing: claim -> stages -> terminal -> ack.

Uses moto SQS and the live database. Each processed message runs in its own session
bound to the test's transaction, so everything rolls back and nothing leaks into the
Spring-shared database.
"""

import json
import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.integrations.s3 import object_exists
from app.integrations.sqs import publish_processing_item, receive_messages
from app.models.enums import RunItemStatus
from app.services.classification import DocumentSection
from app.services.filing import SECTION_TABLES
from app.services.section_specs import SECTION_SPECS
from app.services.source_loading import load_source_document
from app.workers.processor import Outcome, process_message
from app.workers.stagetypes import RejectStageError, StageContext, TransientStageError
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
    db_session,
    make_document,
    status: str = "queued",
    *,
    intended_section: str | None = None,
) -> tuple[uuid.UUID, uuid.UUID, int]:
    document_id = make_document()
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            # source_key mirrors what create_run copies onto the item at submit, which is
            # what the stages load the document through.
            "INSERT INTO ai_processing_run_items "
            "(run_id, document_id, status, intended_section, source_key) "
            "VALUES (:r, :rep, :s, :sec, "
            "(SELECT filepath FROM unclassified_files WHERE id = :rep)) "
            "RETURNING id"
        ),
        {"r": run_id, "rep": document_id, "s": status, "sec": intended_section},
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


def _item(db_session, item_id):
    return (
        db_session.execute(
            text(
                "SELECT section_row_id, filed_section, source_key, last_error_code "
                "FROM ai_processing_run_items WHERE id = :id"
            ),
            {"id": item_id},
        )
        .mappings()
        .one()
    )


def _filed_row(db_session, item_id):
    """The section row this item was filed into, read from whichever table that is."""
    item = _item(db_session, item_id)
    assert item["section_row_id"] is not None, "item was never filed"
    table = SECTION_TABLES[DocumentSection(item["filed_section"])]
    return db_session.execute(select(table).where(table.c.id == item["section_row_id"])).one()


@pytest.mark.parametrize("section", ["insurance", "scans_imaging", "vaccinations"])
def test_a_section_document_is_extracted_and_filed(
    db_session, make_document, session_factory, test_settings, aws, section
):
    """A non-report section runs extract_section, is filed into its own table, and stops.

    No insights stage — there is nothing clinical to interpret — which is exactly why the
    content carries an explicit `state`: null insights here means "never", not "pending".
    """
    _, sqs, queue_url, _ = aws
    # A real PDF: this path reads the document's text rather than sending the file.
    document_id = make_document(body=text_pdf("Policy Period 01/10/2019 to 30/09/2020"))
    item_id, run_id, _ = _seed_item(db_session, lambda **_: document_id)
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

    item = _item(db_session, item_id)
    assert item["filed_section"] == section
    # Filed out of intake, into that section's own table under its own prefix.
    assert not _source_exists(db_session, document_id)
    filed = _filed_row(db_session, item_id)
    assert filed.filepath.startswith(f"{section}/")
    assert item["source_key"] == filed.filepath
    assert filed.content["ai"]["state"] == "complete"
    assert filed.content["ai"]["section_extraction"] is not None
    assert filed.content["ai"]["insights"] is None
    assert _queue_depth(sqs, queue_url) == 0


def test_a_report_is_filed_with_its_extraction_and_insights(
    db_session, make_document, session_factory, test_settings, aws
):
    """No intended section: the detected one wins and the document is filed there."""
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.COMPLETED
    assert _item(db_session, item_id)["filed_section"] == "reports"
    assert not _source_exists(db_session, document_id)
    content = _filed_row(db_session, item_id).content["ai"]
    assert content["state"] == "complete"
    assert content["extraction"] is not None
    assert content["insights"] is not None


def test_an_upload_into_the_matching_section_behaves_identically(
    db_session, make_document, session_factory, test_settings, aws
):
    """The intended section only ever *blocks*; agreeing with it changes nothing.

    Deliberately a non-report section: that is the combination a user actually produces by
    picking a section in the app, and it is the one whose finished content has null
    insights for ever.
    """
    _, sqs, queue_url, _ = aws
    document_id = make_document(body=text_pdf("Vaccine: Tetanus  Next due 14/03/2027"))
    item_id, run_id, _ = _seed_item(
        db_session, lambda **_: document_id, intended_section="vaccinations"
    )
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(
        sqs, queue_url, session_factory, test_settings, aws, ai=_ClassifiesAs("vaccinations")
    )

    assert outcome is Outcome.COMPLETED
    assert _item(db_session, item_id)["filed_section"] == "vaccinations"
    content = _filed_row(db_session, item_id).content["ai"]
    assert content["state"] == "complete"
    assert content["section_extraction"] is not None
    # A section document has no insights, permanently — which is why `state` exists.
    assert content["insights"] is None


def test_a_section_mismatch_stops_at_classification_and_files_nothing(
    db_session, make_document, session_factory, test_settings, aws
):
    """Uploaded into Reports, but it is an insurance policy. Stop before paying for
    extraction, and leave the document where the user can re-file it."""
    _, sqs, queue_url, _ = aws
    s3 = aws[0]
    item_id, run_id, document_id = _seed_item(db_session, make_document, intended_section="reports")
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(
        sqs, queue_url, session_factory, test_settings, aws, ai=_ClassifiesAs("insurance")
    )

    assert outcome is Outcome.REJECTED
    item = _item(db_session, item_id)
    assert item["last_error_code"] == "section_mismatch"
    assert item["section_row_id"] is None
    # Still in intake, object untouched.
    assert _source_exists(db_session, document_id)
    assert object_exists(s3, test_settings.s3_bucket, item["source_key"]) is True
    # Nothing was extracted: the mismatch is caught before the section's stages run.
    assert _section_row(db_session, item_id) is None


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
    item = _item(db_session, item_id)
    assert item["last_error_code"] == section
    assert item["section_row_id"] is None
    assert _source_exists(db_session, document_id)
    # The classification is still recorded even though nothing processed it.
    assert (
        db_session.execute(
            text("SELECT section FROM ai_report_classifications WHERE run_item_id = :id"),
            {"id": item_id},
        ).scalar_one()
        == section
    )


# --- a filed document must never be left saying "still processing" ----------


def test_a_reject_after_filing_leaves_the_document_filed_and_marked_failed(
    db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    """Filing is the primary value; insights are the bonus. A failed extraction must not
    put the document back in Unclassified — but the content must stop saying 'classified'.
    """
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    def _reject(_ctx) -> None:
        raise RejectStageError("extraction_failed", "Unreadable")

    monkeypatch.setattr(
        "app.workers.processor.SECTION_PIPELINES",
        {DocumentSection.REPORTS: [(RunItemStatus.EXTRACTING, _reject)]},
    )

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.REJECTED
    # Filed anyway, and not put back into intake.
    assert _item(db_session, item_id)["section_row_id"] is not None
    assert not _source_exists(db_session, document_id)
    assert _filed_row(db_session, item_id).content["ai"]["state"] == "failed"


def test_a_cancel_after_filing_marks_the_filed_content_failed(
    db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    def _cancelling_stage(ctx) -> None:
        ctx.session.execute(
            text("UPDATE ai_processing_run_items SET status='cancelled' WHERE id=:id"),
            {"id": ctx.item_id},
        )
        ctx.session.commit()

    monkeypatch.setattr(
        "app.workers.processor.SECTION_PIPELINES",
        {
            DocumentSection.REPORTS: [
                (RunItemStatus.EXTRACTING, _cancelling_stage),
                (RunItemStatus.GENERATING_INSIGHTS, _cancelling_stage),
            ]
        },
    )

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.CANCELLED
    assert _status(db_session, item_id) == RunItemStatus.CANCELLED.value
    assert _filed_row(db_session, item_id).content["ai"]["state"] == "failed"


def test_giving_up_marks_a_filed_document_failed(
    db_session, make_document, session_factory, test_settings, aws
):
    """The attempt cap is a terminal path too: `claim_item` marks the item failed and never
    reaches the pipeline, so the filed row's content has to be stamped here."""
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    db_session.execute(
        text(
            "INSERT INTO ai_report_classifications (run_item_id, document_id, section, title, "
            "confidence, prompt_version, schema_version) "
            "VALUES (:i, :d, 'reports', 'CBC', 0.9, 'clf-2', 'clf-2')"
        ),
        {"i": item_id, "d": document_id},
    )
    row_id = db_session.execute(
        text(
            "INSERT INTO reports (user_id, filepath, created_by, content) "
            "SELECT user_id, 'reports/x.pdf', created_by, CAST(:c AS JSONB) "
            "FROM unclassified_files WHERE id = :d RETURNING id"
        ),
        {"d": document_id, "c": json.dumps({"ai": {"state": "classified"}})},
    ).scalar_one()
    db_session.execute(
        text(
            "UPDATE ai_processing_run_items SET attempt_count=:a, status='extracting', "
            "section_row_id=:s, filed_section='reports' WHERE id=:id"
        ),
        {"a": test_settings.max_attempts, "s": row_id, "id": item_id},
    )
    db_session.flush()
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.GAVE_UP
    content = db_session.execute(
        text("SELECT content FROM reports WHERE id = :id"), {"id": row_id}
    ).scalar_one()
    assert content["ai"]["state"] == "failed"


def test_a_transient_failure_after_filing_leaves_the_content_classified(
    db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    """The mirror of the failed-stamping tests: the document is about to be RETRIED, so
    its content must keep saying `classified`. Stamping `failed` here would flash a
    permanent-looking failure at the user for every provider blip."""
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    def _flaky(_ctx) -> None:
        raise TransientStageError("provider 503")

    monkeypatch.setattr(
        "app.workers.processor.SECTION_PIPELINES",
        {DocumentSection.REPORTS: [(RunItemStatus.EXTRACTING, _flaky)]},
    )

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.RETRY
    # Filed — that part is durable and must survive the retry.
    assert _item(db_session, item_id)["section_row_id"] is not None
    assert _filed_row(db_session, item_id).content["ai"]["state"] == "classified"
    # And the message is still on the queue to be retried.
    assert _queue_depth(sqs, queue_url) == 1


def test_a_redelivery_that_re_enters_filing_does_not_file_twice(
    db_session, make_document, session_factory, test_settings, aws
):
    """SQS is at-least-once, and a redelivery re-runs the pipeline **from the top** —
    including filing. The item is deliberately put back into a claimable state first, so
    this actually reaches `file_document` rather than being skipped as terminal (which is
    what `test_duplicate_delivery_of_completed_item_is_a_noop` covers)."""
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)
    _process(sqs, queue_url, session_factory, test_settings, aws)
    first = _item(db_session, item_id)["section_row_id"]
    assert first is not None

    # As a worker that crashed after filing would have left it: filed, but not terminal.
    db_session.execute(
        text(
            "UPDATE ai_processing_run_items "
            "SET status='processing', completed_at=NULL, attempt_count=0 WHERE id=:id"
        ),
        {"id": item_id},
    )
    db_session.flush()
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)
    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.COMPLETED
    # The already-filed early return gave back the same row instead of inserting another —
    # and did not try to re-read the intake row, which the first pass deleted.
    assert _item(db_session, item_id)["section_row_id"] == first
    assert (
        db_session.execute(
            text("SELECT count(*) FROM reports WHERE id = :id"), {"id": first}
        ).scalar_one()
        == 1
    )


# --- retrying a document that was already filed -----------------------------


def test_a_filed_but_failed_document_is_reprocessed_in_place_on_retry(
    api, db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    """The whole point of retry after the filing rewrite: extraction re-runs and the filed
    row's `content` is updated, rather than the document being filed a second time.

    A retry is a *new* run item, so it has no `section_row_id` of its own and the intake row
    the first pass deleted is not coming back. Without adoption the second pass rejects with
    `source_document_missing` and the document is stuck at `state: failed` for ever — the
    API answers 202 and nothing ever finishes.
    """
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    def _reject(_ctx) -> None:
        raise RejectStageError("extraction_failed", "Unreadable")

    monkeypatch.setattr(
        "app.workers.processor.SECTION_PIPELINES",
        {DocumentSection.REPORTS: [(RunItemStatus.EXTRACTING, _reject)]},
    )
    assert _process(sqs, queue_url, session_factory, test_settings, aws) is Outcome.REJECTED
    filed_row_id = _item(db_session, item_id)["section_row_id"]
    assert filed_row_id is not None
    assert _filed_row(db_session, item_id).content["ai"]["state"] == "failed"

    monkeypatch.undo()  # whatever broke extraction is fixed; the retry runs the real pipeline
    response = api.post(f"/v1/documents/reports/{document_id}/ai-result/retry")
    assert response.status_code == 202
    retry_item_id = uuid.UUID(response.json()["item_id"])
    assert retry_item_id != item_id

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.COMPLETED
    assert _status(db_session, retry_item_id) == RunItemStatus.COMPLETED.value
    # The SAME row, adopted and updated in place.
    retry_item = _item(db_session, retry_item_id)
    assert retry_item["section_row_id"] == filed_row_id
    assert retry_item["filed_section"] == "reports"
    # And no second copy of the document: one row in `reports` for that object.
    assert (
        db_session.execute(
            text("SELECT count(*) FROM reports WHERE filepath = :k"),
            {"k": retry_item["source_key"]},
        ).scalar_one()
        == 1
    )
    content = _filed_row(db_session, retry_item_id).content["ai"]
    assert content["state"] == "complete"
    assert content["extraction"] is not None


def test_a_retry_that_classifies_into_another_section_is_rejected(
    api, db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    """A loud terminal reject, not a re-file. Moving the document again would mean deleting
    a Spring row we created and copying the object a second time — a lot of machinery for
    something that should not happen at temperature=0, and data-mangling if it misfires."""
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    def _reject(_ctx) -> None:
        raise RejectStageError("extraction_failed", "Unreadable")

    monkeypatch.setattr(
        "app.workers.processor.SECTION_PIPELINES",
        {DocumentSection.REPORTS: [(RunItemStatus.EXTRACTING, _reject)]},
    )
    _process(sqs, queue_url, session_factory, test_settings, aws)
    filed_row_id = _item(db_session, item_id)["section_row_id"]
    assert filed_row_id is not None

    monkeypatch.undo()
    retry_item_id = uuid.UUID(
        api.post(f"/v1/documents/reports/{document_id}/ai-result/retry").json()["item_id"]
    )

    outcome = _process(
        sqs, queue_url, session_factory, test_settings, aws, ai=_ClassifiesAs("insurance")
    )

    assert outcome is Outcome.REJECTED
    retry_item = _item(db_session, retry_item_id)
    assert retry_item["last_error_code"] == "section_changed_on_retry"
    assert retry_item["section_row_id"] is None
    # Neither table gained a row: the original stands, and nothing was filed into insurance.
    key = {"k": retry_item["source_key"]}
    assert (
        db_session.execute(text("SELECT count(*) FROM reports WHERE filepath = :k"), key)
    ).scalar_one() == 1
    assert (
        db_session.execute(text("SELECT count(*) FROM insurance WHERE filepath = :k"), key)
    ).scalar_one() == 0


# --- the stages must survive the intake row disappearing --------------------


def test_stages_load_the_document_after_the_intake_row_is_gone(
    db_session, make_document, test_settings, aws
):
    """The pipeline must not depend on unclassified_files: filing deletes that row
    mid-pipeline, and extraction still has to fetch the document afterwards."""
    document_id = make_document(key="reports/a1b2.pdf")
    item_id, run_id, _ = _seed_item(db_session, lambda **_: document_id)
    db_session.execute(text("DELETE FROM unclassified_files WHERE id = :id"), {"id": document_id})
    db_session.flush()

    ctx = StageContext(
        item_id=item_id,
        run_id=run_id,
        document_id=document_id,
        source_key="reports/a1b2.pdf",
        attempt=1,
        session=db_session,
        s3=aws[0],
        ai=_FAKE_AI,
        settings=test_settings,
    )

    payload = load_source_document(ctx)

    assert payload.filename == "reports/a1b2.pdf"


def test_a_missing_source_key_is_a_permanent_reject(db_session, make_document, test_settings, aws):
    """Every item gets a source_key at submit, so a gap is an anomaly — never retried."""
    document_id = make_document()
    item_id, run_id, _ = _seed_item(db_session, lambda **_: document_id)

    ctx = StageContext(
        item_id=item_id,
        run_id=run_id,
        document_id=document_id,
        source_key="",
        attempt=1,
        session=db_session,
        s3=aws[0],
        ai=_FAKE_AI,
        settings=test_settings,
    )

    with pytest.raises(RejectStageError) as excinfo:
        load_source_document(ctx)
    assert excinfo.value.code == "source_document_missing"
