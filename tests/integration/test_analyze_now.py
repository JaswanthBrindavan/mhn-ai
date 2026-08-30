"""The user's "analyse this now" choice, made at upload time.

Three properties, and two of them are about NOT spending money.

It opts out of the ANALYSIS_ON_DEMAND pause for one document. It is per SUBMISSION and
never inherited — a reassigned document belongs to somebody who has asked for nothing, and
the uploader's tick must not spend on their records. And it cannot override the global flag
being off, because off is the emergency stop and already means "everything runs fully".
"""

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.integrations.sqs import publish_processing_item, receive_messages
from app.workers.processor import Outcome, process_message
from tests.support.ai import FakeAIProvider

pytestmark = pytest.mark.integration

_FAKE_AI = FakeAIProvider()


@pytest.fixture
def session_factory(db_connection):
    """Fresh sessions on the test connection, so worker commits stay inside the roll-back."""

    def _make() -> Session:
        return Session(bind=db_connection, join_transaction_mode="create_savepoint")

    return _make


def _submit(api, document_id: int, *, analyze_now: bool | None = None):
    document: dict[str, object] = {"document_id": document_id}
    if analyze_now is not None:
        document["analyze_now"] = analyze_now
    return api.post("/v1/document-processing-runs", json={"documents": [document]})


def _items(db_session, document_id: int):
    """Every item for a document. Not "the latest": every insert in this suite runs inside
    ONE transaction and Postgres's now() is the TRANSACTION time, so two attempts tie on
    created_at and "latest" is a coin flip."""
    return (
        db_session.execute(
            text(
                "SELECT id, status, analyze_now FROM ai_processing_run_items WHERE document_id = :d"
            ),
            {"d": document_id},
        )
        .mappings()
        .all()
    )


def _item(db_session, document_id: int):
    rows = _items(db_session, document_id)
    assert len(rows) == 1, f"expected one item, found {len(rows)}"
    return rows[0]


def test_the_choice_is_stored_on_the_item(api, db_session, make_document) -> None:
    document_id = make_document()

    assert _submit(api, document_id, analyze_now=True).status_code == 202

    assert _item(db_session, document_id)["analyze_now"] is True


def test_a_submission_that_says_nothing_defaults_to_the_pause(
    api, db_session, make_document
) -> None:
    """Every caller that has never heard of this field keeps today's behaviour."""
    document_id = make_document()

    assert _submit(api, document_id).status_code == 202

    assert _item(db_session, document_id)["analyze_now"] is False


def test_a_resubmission_does_not_inherit_the_choice(api, db_session, make_document) -> None:
    """**The reassign rule, and the whole reason the column lives on the run item.**

    Spring's reassign re-submits the same document for a NEW owner, who has asked for
    nothing. A run item is per attempt and inherits nothing, so the second submission
    starts false — without any code reaching in to reset it.
    """
    document_id = make_document()
    _submit(api, document_id, analyze_now=True)
    first = _item(db_session, document_id)
    # The first attempt has to be finished, or the second submission REUSES it.
    db_session.execute(
        text("UPDATE ai_processing_run_items SET status = 'rejected' WHERE id = :i"),
        {"i": first["id"]},
    )
    db_session.commit()

    _submit(api, document_id)  # the reassign: no tick, because nobody ticked one

    rows = _items(db_session, document_id)
    assert len(rows) == 2, "expected a second attempt, not a reuse"
    second = next(r for r in rows if r["id"] != first["id"])
    assert second["analyze_now"] is False


def test_reusing_an_active_item_clears_a_choice_the_new_submission_did_not_make(
    api, db_session, make_document
) -> None:
    """The one hole in "a new row defaults it away": there is no new row.

    Reachable exactly as a user reaches it — a mismatched document is retried, so an active
    item carries analyze_now=true, and the user then answers "it's my father's" in the
    dialog that is still open. Without this the in-flight item keeps the uploader's tick and
    analyses a document that is about to become somebody else's.
    """
    document_id = make_document()
    _submit(api, document_id, analyze_now=True)
    active = _item(db_session, document_id)
    assert active["analyze_now"] is True

    _submit(api, document_id)  # reused, not created — the item is still active

    after = _item(db_session, document_id)
    assert after["id"] == active["id"], "expected the active item to be reused"
    assert after["analyze_now"] is False


def test_reusing_an_active_item_never_turns_the_choice_on(api, db_session, make_document) -> None:
    """Only ever clears. Setting it here would let a second submission spend money on a
    document whose first submission did not ask for it — and a reassign is that shape."""
    document_id = make_document()
    _submit(api, document_id)
    active = _item(db_session, document_id)
    assert active["analyze_now"] is False

    _submit(api, document_id, analyze_now=True)

    after = _item(db_session, document_id)
    assert after["id"] == active["id"]
    assert after["analyze_now"] is False


def test_an_unknown_field_is_still_refused(api, make_document) -> None:
    """`extra="forbid"` stays — adding a field must not quietly widen the contract."""
    response = api.post(
        "/v1/document-processing-runs",
        json={"documents": [{"document_id": make_document(), "analyse_now": True}]},
    )

    assert response.status_code == 422


# --- what the worker does with it -------------------------------------------


def _run(sqs, queue_url, session_factory, settings, aws):
    messages = receive_messages(
        sqs, queue_url, max_messages=1, wait_seconds=0, visibility_timeout=30
    )
    assert len(messages) == 1
    return process_message(
        messages[0],
        session_factory=session_factory,
        s3=aws[0],
        sqs=sqs,
        ai=_FAKE_AI,
        settings=settings,
    )


def _seed_queued(db_session, document_id: int, *, analyze_now: bool):
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    key = db_session.execute(
        text("SELECT filepath FROM unclassified_files WHERE id = :d"), {"d": document_id}
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items "
            "(run_id, document_id, status, source_key, analyze_now) "
            "VALUES (:r, :d, 'queued', :k, :a) RETURNING id"
        ),
        {"r": run_id, "d": document_id, "k": key, "a": analyze_now},
    ).scalar_one()
    db_session.commit()
    return item_id, run_id


def _content_state(db_session, item_id) -> str:
    row = db_session.execute(
        text(
            "SELECT r.content FROM reports r JOIN ai_processing_run_items i "
            "ON i.section_row_id = r.id WHERE i.id = :i"
        ),
        {"i": item_id},
    ).scalar_one()
    return str(row["ai"]["state"])


def test_the_pipeline_runs_in_full_when_the_user_asked(
    db_session, make_document, session_factory, test_settings, aws
) -> None:
    _, sqs, queue_url, _ = aws
    document_id = make_document()
    item_id, run_id = _seed_queued(db_session, document_id, analyze_now=True)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)
    # On-demand ON, which is the default and the only state where the tick means anything.
    settings = test_settings.model_copy(update={"analysis_on_demand": True})

    outcome = _run(sqs, queue_url, session_factory, settings, aws)

    assert outcome is Outcome.COMPLETED
    assert _content_state(db_session, item_id) == "complete"


def test_the_pipeline_still_pauses_when_the_user_did_not(
    db_session, make_document, session_factory, test_settings, aws
) -> None:
    _, sqs, queue_url, _ = aws
    document_id = make_document()
    item_id, run_id = _seed_queued(db_session, document_id, analyze_now=False)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)
    settings = test_settings.model_copy(update={"analysis_on_demand": True})

    outcome = _run(sqs, queue_url, session_factory, settings, aws)

    assert outcome is Outcome.COMPLETED
    assert _content_state(db_session, item_id) == "classified"


def test_the_choice_cannot_override_the_emergency_stop(
    db_session, make_document, session_factory, test_settings, aws
) -> None:
    """ANALYSIS_ON_DEMAND off already means everything runs fully, so a tick buys nothing
    and — more to the point — cannot be used to route around the stop."""
    _, sqs, queue_url, _ = aws
    document_id = make_document()
    item_id, run_id = _seed_queued(db_session, document_id, analyze_now=True)
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)
    settings = test_settings.model_copy(update={"analysis_on_demand": False})

    outcome = _run(sqs, queue_url, session_factory, settings, aws)

    assert outcome is Outcome.COMPLETED
    assert _content_state(db_session, item_id) == "complete"
