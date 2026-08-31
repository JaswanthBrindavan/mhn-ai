"""End-to-end message processing: claim -> stages -> terminal -> ack.

Uses moto SQS and the live database. Each processed message runs in its own session
bound to the test's transaction, so everything rolls back and nothing leaks into the
Spring-shared database.
"""

import json
import urllib.error
import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.integrations.s3 import object_exists
from app.integrations.sqs import publish_processing_item, receive_messages
from app.models.enums import RunItemStatus
from app.services import filing, notify
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


# --- on-demand analysis -----------------------------------------------------


def _on_demand(test_settings):
    return test_settings.model_copy(update={"analysis_on_demand": True})


def test_a_document_being_read_says_so_rather_than_looking_paused(
    db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    """`classified` was answering two questions with one word.

    A document uploaded with "read it straight away" ticked is filed with `classified`
    and then analysed immediately, so for the whole 30-80s the stages take it looked
    identical to one waiting to be asked about — and the app put an "Analyse document"
    button in front of work already under way. Pressing it returned 409
    already_in_progress.

    Asserted from INSIDE a stage, because that is the only moment the value exists: by
    the time the pipeline returns, the row says `complete`.
    """
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    # The tick: "read it straight away". This is the case that looked paused — a document
    # nobody has to press anything for, which the app was asking them to press for.
    db_session.execute(
        text("UPDATE ai_processing_run_items SET analyze_now = true WHERE id = :i"),
        {"i": item_id},
    )
    db_session.commit()
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    seen: dict[str, str] = {}

    def _peek(ctx) -> None:
        seen["state"] = ctx.session.execute(
            text(
                "SELECT content FROM reports WHERE id = "
                "(SELECT section_row_id FROM ai_processing_run_items WHERE id = :i)"
            ),
            {"i": ctx.item_id},
        ).scalar_one()["ai"]["state"]

    monkeypatch.setattr(
        "app.workers.processor.SECTION_PIPELINES",
        {DocumentSection.REPORTS: [(RunItemStatus.EXTRACTING, _peek)]},
    )

    outcome = _process(sqs, queue_url, session_factory, _on_demand(test_settings), aws)

    assert outcome is Outcome.COMPLETED
    assert seen["state"] == "analysing"
    # And it does not stay that way: the pipeline finished, so the row is terminal.
    assert _filed_row(db_session, item_id).content["ai"]["state"] == "complete"


def test_a_paused_document_is_not_marked_as_being_read(
    db_session, make_document, session_factory, test_settings, aws
):
    """The other half of the same distinction, and the one the flag exists for.

    Nothing is running, so the row must keep saying `classified` — that is what puts the
    Analyse button on screen at all.
    """
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    _process(sqs, queue_url, session_factory, _on_demand(test_settings), aws)

    assert _filed_row(db_session, item_id).content["ai"]["state"] == "classified"


def test_analysis_on_demand_files_the_document_and_stops(
    db_session, make_document, session_factory, test_settings, aws
):
    """The point of the flag: visible in seconds, nothing expensive spent."""
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(sqs, queue_url, session_factory, _on_demand(test_settings), aws)

    assert outcome is Outcome.COMPLETED
    assert _status(db_session, item_id) == RunItemStatus.COMPLETED.value

    # Filed — the document is in its section and openable.
    section_row_id = db_session.execute(
        text("SELECT section_row_id FROM ai_processing_run_items WHERE id = :i"), {"i": item_id}
    ).scalar_one()
    assert section_row_id is not None

    # ...and marked as read no further. `classified` already means exactly this, which is
    # why the pause needed no new state anywhere.
    content = db_session.execute(
        text("SELECT content FROM reports WHERE id = :r"), {"r": section_row_id}
    ).scalar_one()
    assert content["ai"]["state"] == "classified"

    # Nothing after classification ran.
    for table in ("ai_report_extractions", "ai_report_insights"):
        assert (
            db_session.execute(
                text(f"SELECT count(*) FROM {table} WHERE run_item_id = :i"), {"i": item_id}
            ).scalar_one()
            == 0
        )
    assert _queue_depth(sqs, queue_url) == 0


def test_with_the_flag_off_the_pipeline_is_exactly_what_it_was(
    db_session, make_document, session_factory, test_settings, aws
):
    """The regression guard that matters most.

    This ships to production before the app has a button, so off must be indistinguishable
    from the code that came before it.
    """
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.COMPLETED
    section_row_id = db_session.execute(
        text("SELECT section_row_id FROM ai_processing_run_items WHERE id = :i"), {"i": item_id}
    ).scalar_one()
    content = db_session.execute(
        text("SELECT content FROM reports WHERE id = :r"), {"r": section_row_id}
    ).scalar_one()
    assert content["ai"]["state"] == "complete"
    assert (
        db_session.execute(
            text("SELECT count(*) FROM ai_report_extractions WHERE run_item_id = :i"),
            {"i": item_id},
        ).scalar_one()
        == 1
    )


def _resume(db_session, sqs, queue_url, document_id):
    """A second run item for a document a previous one already filed, as create_run makes
    it: new item, no section_row_id of its own, source_key copied from the filed one."""
    prior = db_session.execute(
        text(
            "SELECT source_key, intended_section FROM ai_processing_run_items "
            "WHERE document_id = :d ORDER BY created_at DESC LIMIT 1"
        ),
        {"d": document_id},
    ).one()
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status, source_key, "
            "intended_section) VALUES (:r, :d, 'queued', :k, :i) RETURNING id"
        ),
        {"r": run_id, "d": document_id, "k": prior.source_key, "i": prior.intended_section},
    ).scalar_one()
    db_session.commit()
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)
    return item_id


def _stages(db_session, item_id):
    return [
        r[0]
        for r in db_session.execute(
            text("SELECT stage FROM ai_process_logs WHERE run_item_id = :i"), {"i": item_id}
        ).all()
    ]


def test_a_resumed_document_is_not_read_a_second_time(
    db_session, make_document, session_factory, test_settings, aws
):
    """Not merely a saved model call.

    A second reading can land on a different section, and _adopt_prior_filing then refuses
    with section_changed_on_retry -- terminal, deliberately not re-filed, and reached by a
    user who did nothing but press a button.
    """
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)
    _process(sqs, queue_url, session_factory, _on_demand(test_settings), aws)

    # Everything the first pass established, so the comparison below is not a row of
    # nulls agreeing with a row of nulls. The identity stamp matters most: dropping it
    # would ask a user who already claimed this document whose it is, all over again --
    # the exact bug identity._carry_settled exists to prevent, through another door.
    db_session.execute(
        text(
            "UPDATE ai_report_classifications SET document_date = DATE '2026-03-12', "
            "document_date_label = 'Sample Collected', patient_name = 'PRIYA MENON', "
            "name_match = 'mismatch', identity_confirmed_at = now() WHERE run_item_id = :i"
        ),
        {"i": item_id},
    )
    db_session.commit()

    resumed_id = _resume(db_session, sqs, queue_url, document_id)
    outcome = _process(sqs, queue_url, session_factory, _on_demand(test_settings), aws)

    assert outcome is Outcome.COMPLETED
    # No model was called to classify: the stage adopted, and ai_process_logs is the
    # record of what was spent, so there is no row for it.
    assert "classifying" not in _stages(db_session, resumed_id)

    columns = (
        "section, title, confidence, document_date, document_date_label, patient_name, "
        "name_match, identity_confirmed_at, prompt_version, schema_version"
    )
    rows = {
        which: db_session.execute(
            text(f"SELECT {columns} FROM ai_report_classifications WHERE run_item_id = :i"),
            {"i": which_id},
        ).one()
        for which, which_id in (("adopted", resumed_id), ("original", item_id))
    }
    assert rows["adopted"] == rows["original"]
    # Spelt out, because the equality above passes on two rows of nulls if the seed ever
    # stops seeding.
    assert rows["adopted"].document_date is not None
    assert rows["adopted"].identity_confirmed_at is not None


def test_resuming_reads_the_document_and_updates_the_same_row(
    db_session, make_document, session_factory, test_settings, aws
):
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)
    _process(sqs, queue_url, session_factory, _on_demand(test_settings), aws)
    filed_row_id = db_session.execute(
        text("SELECT section_row_id FROM ai_processing_run_items WHERE id = :i"), {"i": item_id}
    ).scalar_one()
    before = db_session.execute(text("SELECT count(*) FROM reports")).scalar_one()

    resumed_id = _resume(db_session, sqs, queue_url, document_id)
    _process(sqs, queue_url, session_factory, _on_demand(test_settings), aws)

    # The same row, read in place -- not a second copy of the document.
    assert db_session.execute(text("SELECT count(*) FROM reports")).scalar_one() == before
    assert (
        db_session.execute(
            text("SELECT section_row_id FROM ai_processing_run_items WHERE id = :i"),
            {"i": resumed_id},
        ).scalar_one()
        == filed_row_id
    )
    content = db_session.execute(
        text("SELECT content FROM reports WHERE id = :r"), {"r": filed_row_id}
    ).scalar_one()
    assert content["ai"]["state"] == "complete"


def test_a_resumed_document_does_not_stop_again(
    db_session, make_document, session_factory, test_settings, aws
):
    # The pause is for the upload, not for every pass. Stopping again would make the
    # button do nothing, forever, with no error anywhere.
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)
    _process(sqs, queue_url, session_factory, _on_demand(test_settings), aws)

    resumed_id = _resume(db_session, sqs, queue_url, document_id)
    _process(sqs, queue_url, session_factory, _on_demand(test_settings), aws)

    assert (
        db_session.execute(
            text("SELECT count(*) FROM ai_report_extractions WHERE run_item_id = :i"),
            {"i": resumed_id},
        ).scalar_one()
        == 1
    )


def test_a_first_pass_still_reads_the_document(
    db_session, make_document, session_factory, test_settings, aws
):
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    _process(sqs, queue_url, session_factory, test_settings, aws)

    assert "classifying" in _stages(db_session, item_id)


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


def test_a_permanent_stage_failure_fails_the_item_without_spending_retries(
    db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    """A truncated response fails identically on every attempt, so retrying it only
    pays the bill again. It ends `failed` rather than `rejected`: rejection means the
    document was routed rather than processed, and Spring is told not to show that as
    an error."""
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    from app.models.enums import RunItemStatus as S
    from app.workers.stages import PermanentStageError

    def _truncated(_ctx) -> None:
        raise PermanentStageError("response_truncated", "hit the output token ceiling")

    monkeypatch.setattr("app.workers.processor.CLASSIFY_STAGE", (S.CLASSIFYING, _truncated))

    outcome = _process(sqs, queue_url, session_factory, test_settings, aws)

    assert outcome is Outcome.FAILED
    assert _status(db_session, item_id) == S.FAILED.value
    row = db_session.execute(
        text("SELECT last_error_code, attempt_count FROM ai_processing_run_items WHERE id=:id"),
        {"id": item_id},
    ).one()
    assert row.last_error_code == "response_truncated"
    # One attempt, not the cap: the point of the change is that the other two are never
    # spent on a failure that cannot come out differently.
    assert row.attempt_count == 1
    assert _queue_depth(sqs, queue_url) == 0  # terminal → dropped, not redelivered


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

    def __init__(self, section: str, handwriting: str = "none") -> None:
        super().__init__()
        self.section = section
        self.handwriting = handwriting

    def analyze_document(self, **kwargs):
        default = super().analyze_document(**kwargs)  # records the call
        if "section" in kwargs["json_schema"].get("properties", {}):
            return structured_response(
                classification_payload(section=self.section, handwriting=self.handwriting)
            )
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


def test_a_section_mismatch_files_where_the_user_put_it_and_reads_nothing(
    db_session, make_document, session_factory, test_settings, aws
):
    """Uploaded into Reports, but it is an insurance policy.

    Their choice wins on WHERE, ours wins on WHETHER. The document goes to Reports because
    that is where they filed it; the pipeline does not run, because reading an insurance
    policy with the report extractor produces confident nonsense.

    **This test used to assert the opposite** — that nothing was filed and the document
    stayed in intake "where the user can re-file it". That recovery was the trap: Spring's
    `moveUnclassified` publishes nothing and deletes the source object, so re-filing by hand
    was the one action that made a document permanently unprocessable. Rewritten rather
    than deleted, because it is the test that pins the decision either way.
    """
    _, sqs, queue_url, _ = aws
    s3 = aws[0]
    item_id, run_id, document_id = _seed_item(db_session, make_document, intended_section="reports")
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(
        sqs, queue_url, session_factory, test_settings, aws, ai=_ClassifiesAs("insurance")
    )

    # Still `rejected` / `section_mismatch`: routing, not an error, and Spring is told not
    # to surface it as one. What is new is that it was filed while being rejected.
    assert outcome is Outcome.REJECTED
    item = _item(db_session, item_id)
    assert item["last_error_code"] == "section_mismatch"
    assert item["filed_section"] == "reports"
    assert item["section_row_id"] is not None
    # Filed like any other document: out of intake, object relocated under the section.
    assert not _source_exists(db_session, document_id)
    assert item["source_key"].startswith("reports/")
    assert object_exists(s3, test_settings.s3_bucket, item["source_key"]) is True
    # And read: nothing. The payload carries the disagreement instead of fields.
    row = _section_row(db_session, item_id)
    assert row["data"]["fields"] == {}
    assert [f["code"] for f in row["data"]["flags"]] == ["section_mismatch"]
    assert "insurance document" in row["data"]["flags"][0]["detail"]


def test_a_mismatch_into_a_section_we_cannot_file_stays_in_intake(
    db_session, make_document, session_factory, test_settings, aws
):
    """`medical_condition` has no table binding here, so there is nowhere to put it.
    Unchanged behaviour: rejected, still in intake, Spring keeps its own mover."""
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(
        db_session, make_document, intended_section="medical_condition"
    )
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(
        sqs, queue_url, session_factory, test_settings, aws, ai=_ClassifiesAs("insurance")
    )

    assert outcome is Outcome.REJECTED
    item = _item(db_session, item_id)
    assert item["last_error_code"] == "section_mismatch"
    assert item["section_row_id"] is None
    assert _source_exists(db_session, document_id)


@pytest.mark.parametrize("section", ["medical_condition", "unknown"])
def test_a_section_with_no_pipeline_is_rejected_by_the_router(
    db_session, make_document, session_factory, test_settings, aws, section
):
    """Routing, not failure: no extractor exists, so the document stays where it is.

    ``prescriptions`` used to be parametrized in here, on the strength of
    ``PRESCRIPTIONS_ENABLED`` defaulting to False. It defaults True as of 2026-08-18, so
    the flag's off-path is now asserted explicitly below rather than riding on a default —
    which is the better test anyway: it names the flag it depends on.
    """
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


def test_turning_the_prescriptions_flag_off_rejects_and_leaves_it_in_intake(
    db_session, make_document, session_factory, test_settings, aws
):
    """The flag's off-path, asserted explicitly now that the default is on.

    Worth keeping even though nothing ships with it off: it is the emergency stop, and this
    is what pressing it does. Note the last assertion — the document stays in
    ``unclassified_files``. Since Spring routes prescriptions through intake, that means a
    prescription the user filed into Prescriptions is left sitting in Unclassified, which is
    why ``config.py`` points at Spring's ``PROCESSABLE`` as the graceful switch instead.
    """
    _, sqs, queue_url, _ = aws
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(
        sqs,
        queue_url,
        session_factory,
        test_settings.model_copy(update={"prescriptions_enabled": False}),
        aws,
        ai=_ClassifiesAs("prescriptions"),
    )

    assert outcome is Outcome.REJECTED
    assert _status(db_session, item_id) == RunItemStatus.REJECTED.value
    item = _item(db_session, item_id)
    # Indistinguishable from a section with no pipeline at all — the detected section is
    # the reason, so a caller can route on it.
    assert item["last_error_code"] == "prescriptions"
    assert item["section_row_id"] is None
    assert _source_exists(db_session, document_id)


def test_a_prescription_is_filed_and_extracted_once_the_flag_is_on(
    db_session, make_document, session_factory, test_settings, aws
):
    """The mirror of the case above: ``PRESCRIPTIONS_ENABLED`` is the only difference.

    The flag is set explicitly rather than left to the default, which is on since
    2026-08-18. A test that depends on a flag should name it, so that flipping the default
    again changes what these two assert rather than which of them silently passes.
    """
    _, sqs, queue_url, _ = aws
    # The stage verifies every name against the document's own text, so the page has to
    # actually print the medicine the model claims to have read off it.
    document_id = make_document(body=text_pdf("Tab. DOLO 650  1-0-1 after food  5 days"))
    item_id, run_id, _ = _seed_item(db_session, lambda **_: document_id)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(
        sqs,
        queue_url,
        session_factory,
        test_settings.model_copy(update={"prescriptions_enabled": True}),
        aws,
        ai=_ClassifiesAs("prescriptions"),
    )

    assert outcome is Outcome.COMPLETED
    row = _section_row(db_session, item_id)
    assert row is not None and row["section"] == "prescriptions"
    assert [m["name_clean"] for m in row["data"]["fields"]["medicines"]] == ["DOLO"]

    item = _item(db_session, item_id)
    assert item["filed_section"] == "prescriptions"
    # Out of intake and into Spring's own prescriptions table, under its own prefix.
    assert not _source_exists(db_session, document_id)
    filed = _filed_row(db_session, item_id)
    assert filed.filepath.startswith("prescriptions/")
    assert filed.content["ai"]["state"] == "complete"
    assert filed.content["ai"]["section_extraction"] is not None
    # Transcription only: there is no insights stage for a prescription.
    assert filed.content["ai"]["insights"] is None


def test_a_handwritten_prescription_is_filed_but_never_read(
    db_session, make_document, session_factory, test_settings, aws
):
    """The one document we deliberately decline to extract.

    A handwritten page has no text layer, so the name guard has nothing to reject with —
    the document most likely to be misread is the one where every downstream check is
    blind. So it is filed (it is the user's prescription and belongs in their section) and
    nothing is read off it; the app asks for the printed pharmacy bill instead.

    Filed, `completed`, and zero medicines: the three things that must all hold at once.
    """
    _, sqs, queue_url, _ = aws
    document_id = make_document(body=text_pdf("Dr A Sharma  Rx  (handwritten)"))
    item_id, run_id, _ = _seed_item(db_session, lambda **_: document_id)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(
        sqs,
        queue_url,
        session_factory,
        test_settings.model_copy(update={"prescriptions_enabled": True}),
        aws,
        ai=_ClassifiesAs("prescriptions", handwriting="mostly"),
    )

    assert outcome is Outcome.COMPLETED
    row = _section_row(db_session, item_id)
    assert row is not None and row["section"] == "prescriptions"
    # Nothing was read off the page.
    assert row["data"]["fields"]["medicines"] == []
    assert [f["code"] for f in row["data"]["flags"]] == ["handwritten_not_extracted"]

    # Still filed, and finished rather than failed — "we did not read this" is an outcome,
    # not an error, and a `failed` state would show the user a bug that isn't one.
    item = _item(db_session, item_id)
    assert item["filed_section"] == "prescriptions"
    assert not _source_exists(db_session, document_id)
    filed = _filed_row(db_session, item_id)
    assert filed.filepath.startswith("prescriptions/")
    assert filed.content["ai"]["state"] == "complete"


def test_a_printed_prescription_is_still_extracted(
    db_session, make_document, session_factory, test_settings, aws
):
    """The guard against over-refusing: only `mostly` stops extraction.

    A scanned printed slip has no text layer either, and must NOT be treated as
    handwritten — extraction is vision, so a missing text layer is no reason to read less.
    """
    _, sqs, queue_url, _ = aws
    document_id = make_document(body=text_pdf("Tab. DOLO 650  1-0-1 after food  5 days"))
    item_id, run_id, _ = _seed_item(db_session, lambda **_: document_id)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(
        sqs,
        queue_url,
        session_factory,
        test_settings.model_copy(update={"prescriptions_enabled": True}),
        aws,
        ai=_ClassifiesAs("prescriptions", handwriting="some"),
    )

    assert outcome is Outcome.COMPLETED
    row = _section_row(db_session, item_id)
    assert [m["name_clean"] for m in row["data"]["fields"]["medicines"]] == ["DOLO"]


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
    its content must NOT say `failed`. Stamping that here would flash a
    permanent-looking failure at the user for every provider blip.

    It says `analysing` rather than `classified`, and that is the honest answer: the
    stages started, one blipped, and a redelivery is coming. `classified` would claim
    nobody is working on it and put an "Analyse document" button in front of a user for
    a document that is mid-flight — which is the bug `ContentState.ANALYSING` exists to
    fix. Nothing is stranded either way: the attempt cap ends the item `failed` through
    `mark_content_failed`, and the retry endpoint is the recovery path from there."""
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
    state = _filed_row(db_session, item_id).content["ai"]["state"]
    assert state == "analysing"
    assert state != "failed"  # spelt out: this is what the test is really about
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


def test_a_retry_of_a_filed_document_cannot_flip_its_section(
    api, db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    """This used to be a loud terminal reject. It is now unreachable, on purpose.

    ``section_changed_on_retry`` existed because a retry re-read the document, and a second
    reading could land somewhere else — leaving the user with a document that could not be
    processed and could not be moved. Re-filing was never the answer: it would mean
    deleting a Spring row we created and copying the object again.

    The answer was to stop re-reading. A filed document has already been classified, and
    the classification is adopted rather than made afresh, so there is nothing for a second
    reading to disagree with. The fake below classifies as insurance and is never asked.

    ``_adopt_prior_filing``'s guard stays as an invariant check, not because this path can
    reach it.
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

    assert outcome is Outcome.COMPLETED
    retry_item = _item(db_session, retry_item_id)
    assert retry_item["last_error_code"] is None
    # The same row it was already filed into, read in place.
    assert retry_item["section_row_id"] == filed_row_id
    assert (
        db_session.execute(
            text("SELECT section FROM ai_report_classifications WHERE run_item_id = :i"),
            {"i": retry_item_id},
        ).scalar_one()
        == "reports"
    )
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


# --- telling Spring the document has landed ---------------------------------


def _capture_notifications(monkeypatch) -> list[dict]:
    """Record what the worker would POST to Spring, running the real notify path."""
    sent: list[dict] = []

    class _Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        sent.append(json.loads(request.data))
        return _Response()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
    return sent


def _notifying(test_settings):
    return test_settings.model_copy(
        update={"spring_callback_url": "http://spring.internal/internal/ai/filed"}
    )


def test_filing_a_document_tells_spring_where_it_went(
    db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    """The announcement carries the row the app must navigate to, and nothing else."""
    _, sqs, queue_url, _ = aws
    sent = _capture_notifications(monkeypatch)
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(sqs, queue_url, session_factory, _notifying(test_settings), aws)

    assert outcome is Outcome.COMPLETED
    section_row_id = _item(db_session, item_id)["section_row_id"]
    assert sent == [
        {
            "document_id": document_id,
            "section": "reports",
            "section_row_id": section_row_id,
            # What the row said at the moment it appeared. The stages that follow update
            # it; the announcement is about arriving, not about finishing.
            "state": "classified",
        }
    ]


def test_a_mismatched_document_is_announced_too(
    db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    """It is filed, it is on screen, and it carries an action. Not announcing it would
    leave the user waiting on the one document that has something to ask them."""
    _, sqs, queue_url, _ = aws
    sent = _capture_notifications(monkeypatch)
    item_id, run_id, document_id = _seed_item(db_session, make_document, intended_section="reports")
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(
        sqs,
        queue_url,
        session_factory,
        _notifying(test_settings),
        aws,
        ai=_ClassifiesAs("insurance"),
    )

    assert outcome is Outcome.REJECTED
    assert [n["section"] for n in sent] == ["reports"]  # where the USER put it
    assert sent[0]["state"] == "complete"


def test_nothing_is_announced_when_nothing_was_filed(
    db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    """The guard in `file_document` matched nothing -- cancelled, or filed underneath us.

    No row appeared, so there is nothing to send anyone to. Announcing here would push the
    user at a section holding no such document.
    """
    _, sqs, queue_url, _ = aws
    sent = _capture_notifications(monkeypatch)
    monkeypatch.setattr(filing, "file_document", lambda *a, **k: None)
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(sqs, queue_url, session_factory, _notifying(test_settings), aws)

    assert outcome is Outcome.CANCELLED
    assert sent == []


def test_the_announcement_happens_after_filing_returns(
    db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    """Ordering, which is the whole correctness argument for where this call sits.

    `file_document` commits as its last database action, so calling after it returns is
    calling after the commit -- and Spring can SELECT the row the moment it is told. The
    reverse order is the same defect as publishing to SQS before committing: the fetch that
    follows finds nothing and the screen sticks.

    **What this proves and what it does not.** It pins the call site relative to filing.
    It cannot prove the commit itself, because the whole suite runs inside one rolled-back
    transaction on one connection, so a second session would see uncommitted rows anyway --
    a test asserting otherwise would pass whatever the order was.
    """
    _, sqs, queue_url, _ = aws
    order: list[str] = []
    real_file = filing.file_document

    def tracking_file(*args, **kwargs):
        result = real_file(*args, **kwargs)
        order.append("filed")
        return result

    class _Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        order.append("announced")
        return _Response()

    monkeypatch.setattr(filing, "file_document", tracking_file)
    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    _process(sqs, queue_url, session_factory, _notifying(test_settings), aws)

    assert order == ["filed", "announced"]


def test_a_failed_announcement_changes_nothing(
    db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    """Spring being down must not cost a redelivery and a second run of every paid stage."""
    _, sqs, queue_url, _ = aws

    def refuse(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(notify.urllib.request, "urlopen", refuse)
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    outcome = _process(sqs, queue_url, session_factory, _notifying(test_settings), aws)

    assert outcome is Outcome.COMPLETED
    assert _item(db_session, item_id)["filed_section"] == "reports"
    assert _filed_row(db_session, item_id).content["ai"]["state"] == "complete"
    assert _queue_depth(sqs, queue_url) == 0  # acked, not left for redelivery


def test_no_callback_url_means_no_call(
    db_session, make_document, session_factory, test_settings, aws, monkeypatch
):
    """Every deployment that has not set the URL behaves exactly as it did before."""
    _, sqs, queue_url, _ = aws

    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("notified with no callback URL configured")

    monkeypatch.setattr(notify.urllib.request, "urlopen", explode)
    item_id, run_id, document_id = _seed_item(db_session, make_document)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)

    assert _process(sqs, queue_url, session_factory, test_settings, aws) is Outcome.COMPLETED
