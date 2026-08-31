"""The gate stops a document BEFORE filing, which is the whole design.

A rejected document keeps its intake row and its object, so the delete path is one row
and one key — there is no Spring section row to unpick, and filing is never undone.
"""

import uuid
from dataclasses import replace

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.integrations.sqs import publish_processing_item, receive_messages
from app.models.enums import RunItemStatus
from app.services.assembly import ContentState, build_content
from app.services.classification import DocumentSection
from app.services.identity import confirm_identity, gate, settled_verdict
from app.services.names import NameVerdict
from app.workers.processor import Outcome, process_message
from app.workers.stagetypes import RejectStageError, StageContext
from tests.support.ai import FakeAIProvider, classification_payload, structured_response

pytestmark = pytest.mark.integration


@pytest.fixture
def gate_ctx(db_session, make_document, aws, test_settings):
    """A `StageContext` for a document that has just been classified.

    `account_name=None` means the intake row is gone — the retry-of-a-filed-document case,
    which the gate must pass rather than relabel as an identity problem.
    """

    def _make(
        *,
        document_name: str | None,
        account_name: str | None,
        enabled: bool = True,
    ) -> StageContext:
        document_id = make_document()
        if account_name is None:
            db_session.execute(
                text("DELETE FROM unclassified_files WHERE id = :d"), {"d": document_id}
            )
        else:
            db_session.execute(
                text(
                    'UPDATE "user" SET name = :n WHERE id = '
                    "(SELECT user_id FROM unclassified_files WHERE id = :d)"
                ),
                {"n": account_name, "d": document_id},
            )
        run_id = db_session.execute(
            text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
        ).scalar_one()
        item_id = db_session.execute(
            text(
                "INSERT INTO ai_processing_run_items (run_id, document_id, status) "
                "VALUES (:r, :d, 'classifying') RETURNING id"
            ),
            {"r": run_id, "d": document_id},
        ).scalar_one()
        db_session.execute(
            text(
                "INSERT INTO ai_report_classifications "
                "(run_item_id, document_id, section, title, confidence, patient_name, "
                " prompt_version, schema_version) "
                "VALUES (:i, :d, 'reports', 'A Report', 0.9, :pn, 'clf-1', 'clf-1')"
            ),
            {"i": item_id, "d": document_id, "pn": document_name},
        )
        db_session.commit()
        return StageContext(
            item_id=uuid.UUID(str(item_id)),
            run_id=uuid.UUID(str(run_id)),
            document_id=int(document_id),
            source_key="unclassified/whatever.pdf",
            attempt=1,
            session=db_session,
            s3=aws[0],
            ai=FakeAIProvider(),
            settings=test_settings.model_copy(update={"name_matching_enabled": enabled}),
        )

    return _make


def test_matching_name_passes(gate_ctx) -> None:
    ctx = gate_ctx(document_name="MR RAJESH SHARMA", account_name="Rajesh Sharma")
    gate(ctx, DocumentSection.REPORTS)  # does not raise


def test_absent_name_passes(gate_ctx) -> None:
    """A document with no printed name is processed as the owner's own — with a note."""
    ctx = gate_ctx(document_name=None, account_name="Rajesh Sharma")
    gate(ctx, DocumentSection.REPORTS)


def test_mismatch_rejects_with_name_mismatch(gate_ctx) -> None:
    ctx = gate_ctx(document_name="PRIYA MENON", account_name="Rajesh Sharma")
    with pytest.raises(RejectStageError) as exc:
        gate(ctx, DocumentSection.REPORTS)
    assert exc.value.code == "name_mismatch"


def test_mismatch_records_the_verdict(gate_ctx, db_session) -> None:
    ctx = gate_ctx(document_name="PRIYA MENON", account_name="Rajesh Sharma")
    with pytest.raises(RejectStageError):
        gate(ctx, DocumentSection.REPORTS)
    assert settled_verdict(db_session, ctx.document_id) is NameVerdict.MISMATCH


def test_a_confirmed_document_is_not_asked_again(gate_ctx, db_session) -> None:
    ctx = gate_ctx(document_name="PRIYA MENON", account_name="Rajesh Sharma")
    with pytest.raises(RejectStageError):
        gate(ctx, DocumentSection.REPORTS)
    confirm_identity(db_session, ctx.document_id)
    gate(ctx, DocumentSection.REPORTS)  # passes second time


def _retry_item(db_session, document_id) -> uuid.UUID:
    """A second run item for the same document, carrying the fresh, verdict-less
    classification a retry inserts."""
    retry_run = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    retry_item = db_session.execute(
        text(
            # Terminal, not active: a partial unique index allows one live item per
            # document, and the first one is still `classifying`.
            "INSERT INTO ai_processing_run_items (run_id, document_id, status) "
            "VALUES (:r, :d, 'completed') RETURNING id"
        ),
        {"r": retry_run, "d": document_id},
    ).scalar_one()
    db_session.execute(
        # `now()` is the transaction's timestamp, so the newer row needs an explicit one.
        text(
            "INSERT INTO ai_report_classifications "
            "(run_item_id, document_id, section, title, confidence, patient_name, "
            " prompt_version, schema_version, created_at) "
            "VALUES (:i, :d, 'reports', 'A Report', 0.9, 'PRIYA MENON', 'clf-1', 'clf-1', "
            " now() + interval '1 hour')"
        ),
        {"i": retry_item, "d": document_id},
    )
    db_session.commit()
    return uuid.UUID(str(retry_item))


def test_a_confirmation_survives_a_retrys_fresh_classification(gate_ctx, db_session) -> None:
    """A retry inserts a NEW classification with a null verdict.

    Reading only the newest row would report "nothing settled" and re-interrogate someone
    about a document they have already claimed as their own.
    """
    ctx = gate_ctx(document_name="PRIYA MENON", account_name="Rajesh Sharma")
    with pytest.raises(RejectStageError):
        gate(ctx, DocumentSection.REPORTS)
    confirm_identity(db_session, ctx.document_id)

    _retry_item(db_session, ctx.document_id)

    assert settled_verdict(db_session, ctx.document_id) is NameVerdict.MATCH
    gate(ctx, DocumentSection.REPORTS)  # still not asked again


def test_a_confirmed_retry_carries_the_verdict_onto_its_own_row(gate_ctx, db_session) -> None:
    """The short-circuit never reaches `record_verdict`, so the retry's OWN classification
    row would keep a null verdict — and `content.ai.name_check` is built from that row.

    What lands there is the stored truth, `mismatch` plus its confirmation stamp, not the
    MATCH `settled_verdict` hands the gate so the document may pass. Writing `match` here
    satisfies the same null check and leaves the standing warning just as unrenderable.
    """
    ctx = gate_ctx(document_name="PRIYA MENON", account_name="Rajesh Sharma")
    with pytest.raises(RejectStageError):
        gate(ctx, DocumentSection.REPORTS)
    confirm_identity(db_session, ctx.document_id)
    retry = replace(ctx, item_id=_retry_item(db_session, ctx.document_id))

    gate(retry, DocumentSection.REPORTS)  # passes, and asks nothing

    assert build_content(db_session, retry.item_id, state=ContentState.COMPLETE)["ai"][
        "name_check"
    ] == {"verdict": "mismatch", "document_name": "PRIYA MENON", "confirmed": True}


def test_an_unprinted_name_reaches_the_payload_as_unknown(gate_ctx, db_session) -> None:
    """The quiet counterpart, on the first pass: this one goes through `record_verdict`
    normally, and the card's "no name was printed" note reads the same key."""
    ctx = gate_ctx(document_name=None, account_name="Rajesh Sharma")

    gate(ctx, DocumentSection.REPORTS)

    assert build_content(db_session, ctx.item_id, state=ContentState.CLASSIFIED)["ai"][
        "name_check"
    ] == {"verdict": "unknown", "document_name": None, "confirmed": False}


def test_unfiled_section_skips_the_gate(gate_ctx) -> None:
    """`medical_condition` is rejected for its own reason; identity would be noise."""
    ctx = gate_ctx(document_name="PRIYA MENON", account_name="Rajesh Sharma")
    gate(ctx, DocumentSection.MEDICAL_CONDITION)


def test_missing_intake_row_passes(gate_ctx) -> None:
    """A retry of a filed document: filing deleted the intake row. Filing owns that case."""
    ctx = gate_ctx(document_name="PRIYA MENON", account_name=None)
    gate(ctx, DocumentSection.REPORTS)


def test_disabled_flag_skips_the_gate(gate_ctx) -> None:
    ctx = gate_ctx(document_name="PRIYA MENON", account_name="Rajesh Sharma", enabled=False)
    gate(ctx, DocumentSection.REPORTS)


# --- end to end, with the flag ON -------------------------------------------
#
# The rest of the worker suite already covers the "no printed name" case with the flag on
# its default of TRUE: the canned classification payload prints no name, so every one of
# those documents passes the gate as UNKNOWN. What is left to pin is the matching case,
# and that a mismatch stops the pipeline *before* filing rather than after.


class _ClassifiesAs(FakeAIProvider):
    """Controls what the classification stage answers; canned for every later stage.

    `section` matters to the tests at the end of this file, which turn on whether a
    document was read a SECOND time: handing the second pass a different answer from the
    first is what makes that observable without counting calls.
    """

    def __init__(self, printed_name: str | None, *, section: str = "reports") -> None:
        super().__init__()
        self.printed_name = printed_name
        self.section = section

    def analyze_document(self, **kwargs):  # type: ignore[no-untyped-def, override]
        if "section" in kwargs["json_schema"].get("properties", {}):
            return structured_response(
                classification_payload(patient_name=self.printed_name, section=self.section)
            )
        return super().analyze_document(**kwargs)


@pytest.fixture
def session_factory(db_connection):
    """Fresh sessions on the test connection, so worker commits stay inside the roll-back."""

    def _make() -> Session:
        return Session(bind=db_connection, join_transaction_mode="create_savepoint")

    return _make


def _publish_one(db_session, make_document, aws, *, owner: str) -> tuple[uuid.UUID, int]:
    _, sqs, queue_url, _ = aws
    document_id = make_document()
    db_session.execute(
        text(
            'UPDATE "user" SET name = :n WHERE id = '
            "(SELECT user_id FROM unclassified_files WHERE id = :d)"
        ),
        {"n": owner, "d": document_id},
    )
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status, source_key) "
            "VALUES (:r, :d, 'queued', "
            "(SELECT filepath FROM unclassified_files WHERE id = :d)) RETURNING id"
        ),
        {"r": run_id, "d": document_id},
    ).scalar_one()
    db_session.flush()
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)
    return uuid.UUID(str(item_id)), int(document_id)


def _run(aws, session_factory, test_settings, ai):
    _, sqs, queue_url, _ = aws
    message = receive_messages(
        sqs, queue_url, max_messages=1, wait_seconds=0, visibility_timeout=30
    )[0]
    return process_message(
        message,
        session_factory=session_factory,
        s3=aws[0],
        sqs=sqs,
        ai=ai,
        settings=test_settings,
    )


def test_a_matching_name_processes_exactly_as_before(
    db_session, make_document, session_factory, test_settings, aws
):
    """The regression that matters most: the flag is ON, so the whole pipeline runs through
    the gate. A document carrying the account holder's name must complete and be filed."""
    assert test_settings.name_matching_enabled is True
    item_id, _ = _publish_one(db_session, make_document, aws, owner="Rajesh Sharma")

    outcome = _run(aws, session_factory, test_settings, _ClassifiesAs("MR RAJESH SHARMA"))

    assert outcome is Outcome.COMPLETED
    row = db_session.execute(
        text(
            "SELECT status, section_row_id, filed_section "
            "FROM ai_processing_run_items WHERE id = :id"
        ),
        {"id": item_id},
    ).one()
    assert row.status == RunItemStatus.COMPLETED.value
    assert row.section_row_id is not None
    assert row.filed_section == "reports"


def test_a_mismatched_name_is_rejected_before_anything_is_filed(
    db_session, make_document, session_factory, test_settings, aws
):
    """Nothing to undo: no section row, and the intake row is still there for the app's
    delete. This is why the gate sits before filing rather than after it."""
    item_id, document_id = _publish_one(db_session, make_document, aws, owner="Rajesh Sharma")

    outcome = _run(aws, session_factory, test_settings, _ClassifiesAs("PRIYA MENON"))

    assert outcome is Outcome.REJECTED
    row = db_session.execute(
        text(
            "SELECT status, last_error_code, section_row_id "
            "FROM ai_processing_run_items WHERE id = :id"
        ),
        {"id": item_id},
    ).one()
    assert row.status == RunItemStatus.REJECTED.value
    assert row.last_error_code == "name_mismatch"
    assert row.section_row_id is None
    assert (
        db_session.execute(
            text("SELECT count(*) FROM unclassified_files WHERE id = :id"), {"id": document_id}
        ).scalar_one()
        == 1
    )
    assert settled_verdict(db_session, document_id) is NameVerdict.MISMATCH


# --- the second reading, and the re-submissions that used to cause one ----------------
#
# `ai_process_logs` showed two `classifying` rows for nearly every document in
# production. Not duplicate processing -- the table is uniquely keyed on
# (item, stage, attempt) and upserted -- but two ITEMS, and the first of them was
# `name_mismatch` every time. The gate refuses BEFORE filing, so the document was never
# filed, so the resumption could not adopt the earlier reading and paid to read the
# document again. See docs/FUTURE.md, "A document classified twice".


def _resubmit(db_session, aws, document_id: int) -> uuid.UUID:
    """A second run item for a document whose first pass is terminal.

    What `/confirm-identity`, Spring's `/reassign` and a hand move all do: a new item
    with no `section_row_id` of its own, published down the ordinary path.
    """
    _, sqs, queue_url, _ = aws
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status, source_key) "
            "VALUES (:r, :d, 'queued', "
            "(SELECT filepath FROM unclassified_files WHERE id = :d)) RETURNING id"
        ),
        {"r": run_id, "d": document_id},
    ).scalar_one()
    db_session.commit()
    publish_processing_item(sqs, queue_url, item_id=item_id, run_id=run_id, document_id=document_id)
    return uuid.UUID(str(item_id))


def _readings(db_session, document_id: int) -> int:
    """How many times a model was asked to classify this document.

    `ai_process_logs` is the record of what was SPENT, and adoption deliberately writes
    no row -- so this counts real readings, not passes.
    """
    return db_session.execute(
        text(
            "SELECT count(*) FROM ai_process_logs l "
            "JOIN ai_processing_run_items i ON i.id = l.run_item_id "
            "WHERE l.stage = 'classifying' AND i.document_id = :d"
        ),
        {"d": document_id},
    ).scalar_one()


def _rename_owner(db_session, document_id: int, name: str) -> None:
    """Stand-in for a reassign: the gate joins intake -> user -> name, so changing the
    name the join lands on is indistinguishable from changing which row it lands on."""
    db_session.execute(
        text(
            'UPDATE "user" SET name = :n WHERE id = '
            "(SELECT user_id FROM unclassified_files WHERE id = :d)"
        ),
        {"n": name, "d": document_id},
    )
    db_session.commit()


def test_confirming_an_identity_does_not_read_the_document_again(
    db_session, make_document, session_factory, test_settings, aws
):
    """The doubled `classifying` row, in its commonest shape.

    The user answered a question about WHOSE the document is. Nothing about that answer
    makes the earlier reading wrong, so the second pass adopts it.
    """
    _, document_id = _publish_one(db_session, make_document, aws, owner="Rajesh Sharma")
    assert (
        _run(aws, session_factory, test_settings, _ClassifiesAs("PRIYA MENON")) is Outcome.REJECTED
    )
    assert _readings(db_session, document_id) == 1

    confirm_identity(db_session, document_id)
    second = _resubmit(db_session, aws, document_id)

    outcome = _run(aws, session_factory, test_settings, _ClassifiesAs("PRIYA MENON"))

    assert outcome is Outcome.COMPLETED
    assert _readings(db_session, document_id) == 1, (
        "the document was read a second time; the reading should have been adopted"
    )
    row = db_session.execute(
        text("SELECT filed_section, section_row_id FROM ai_processing_run_items WHERE id = :i"),
        {"i": second},
    ).one()
    assert row.filed_section == "reports"
    assert row.section_row_id is not None


def test_a_reassigned_document_adopts_the_reading_and_recomputes_the_verdict(
    db_session, make_document, session_factory, test_settings, aws
):
    """The trap inside the optimisation, and why it is not a one-liner.

    The name PRINTED on the page never changes; the account name it is compared against
    does. So the reading is adopted and the verdict is worked out afresh -- carrying the
    stored `mismatch` forward would refuse the document in the very wallet it belongs to.

    The second pass is handed a provider that would answer with a different name
    entirely: if anything re-read the document, the verdict below would be MISMATCH.
    """
    _, document_id = _publish_one(db_session, make_document, aws, owner="Rajesh Sharma")
    assert (
        _run(aws, session_factory, test_settings, _ClassifiesAs("PRIYA MENON")) is Outcome.REJECTED
    )
    assert settled_verdict(db_session, document_id) is NameVerdict.MISMATCH

    _rename_owner(db_session, document_id, "Priya Menon")
    second = _resubmit(db_session, aws, document_id)

    outcome = _run(aws, session_factory, test_settings, _ClassifiesAs("SOMEBODY ELSE"))

    assert outcome is Outcome.COMPLETED
    assert _readings(db_session, document_id) == 1
    # This pass's OWN row, rather than `settled_verdict`: every row here is written inside
    # one test transaction, so they share `now()` and the newest-first ordering
    # `settled_row` relies on has nothing to break the tie with. (`_retry_item` above
    # sidesteps the same thing by stamping an explicit timestamp.) It is also the more
    # precise claim -- what this pass concluded, not what some row for the document says.
    row = db_session.execute(
        text(
            "SELECT patient_name, name_match FROM ai_report_classifications WHERE run_item_id = :i"
        ),
        {"i": second},
    ).one()
    assert row.patient_name == "PRIYA MENON", "the printed name should have been adopted"
    assert row.name_match == NameVerdict.MATCH.value


def test_a_document_rejected_as_unknown_is_still_read_again(
    db_session, make_document, session_factory, test_settings, aws
):
    """Adoption is deliberately NOT extended to every terminal state.

    `unknown` means the reading itself failed to place the document, and a retry is
    someone asking for it to be read AGAIN -- usually after a prompt or model change.
    Adopting there would turn the one endpoint that can rescue it into a no-op.
    """
    _, document_id = _publish_one(db_session, make_document, aws, owner="Rajesh Sharma")
    assert (
        _run(aws, session_factory, test_settings, _ClassifiesAs(None, section="unknown"))
        is Outcome.REJECTED
    )
    assert _readings(db_session, document_id) == 1

    _resubmit(db_session, aws, document_id)
    outcome = _run(aws, session_factory, test_settings, _ClassifiesAs(None))

    assert outcome is Outcome.COMPLETED
    assert _readings(db_session, document_id) == 2
