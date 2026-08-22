"""Filing a classified document into its section table.

The ordering under test is the point: S3 copy, then one DB transaction, then the S3 delete.
Any other order can leave a live row pointing at a deleted object.
"""

import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import NamedTuple

import pytest
from sqlalchemy import event, select, text

from app.integrations.s3 import (
    SourceObjectMissingError,
    SourceObjectUnavailableError,
    object_exists,
)
from app.models.spring import bills, reports, unclassified_files, vaccinations
from app.services import filing
from app.services.assembly import ContentState, build_content
from app.services.classification import DocumentSection
from app.services.s3_keys import preview_key_for
from app.workers.stagetypes import RejectStageError, TransientStageError

from .conftest import BUCKET

pytestmark = pytest.mark.integration


class Seeded(NamedTuple):
    document_id: int
    item_id: uuid.UUID
    source_key: str


@pytest.fixture
def s3_client(aws):
    return aws[0]


@pytest.fixture
def bucket() -> str:
    return BUCKET


@pytest.fixture
def classified_item(db_session, aws, make_document):
    """A document seeded under `unclassified/`, with a run item at `classifying` and a
    classification row — the exact state the pipeline is in when it calls the filer.

    The seed is COMMITTED: `file_document` rolls back to a savepoint on its guard-failure
    path, which would otherwise take the seed with it.
    """
    s3 = aws[0]

    def _make(section: DocumentSection, *, with_preview: bool = False) -> Seeded:
        key = f"unclassified/{uuid.uuid4().hex}.pdf"
        document_id = make_document(key=key)
        if with_preview:
            s3.put_object(Bucket=BUCKET, Key=preview_key_for(key), Body=b"preview-bytes")

        run_id = db_session.execute(
            text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
        ).scalar_one()
        item_id = db_session.execute(
            text(
                "INSERT INTO ai_processing_run_items (run_id, document_id, status, source_key) "
                "VALUES (:r, :d, 'classifying', :k) RETURNING id"
            ),
            {"r": run_id, "d": document_id, "k": key},
        ).scalar_one()
        db_session.execute(
            text(
                "INSERT INTO ai_report_classifications (run_item_id, document_id, section, "
                "title, confidence, prompt_version, schema_version) "
                "VALUES (:i, :d, :s, 'Seed Document', 0.95, 'clf-2', 'clf-2')"
            ),
            {"i": item_id, "d": document_id, "s": section.value},
        )
        db_session.commit()
        return Seeded(document_id=document_id, item_id=item_id, source_key=key)

    return _make


@pytest.fixture
def seed_section_extraction(db_session):
    def _seed(item_id, document_id, section: str, fields: dict, flags: list | None = None) -> None:
        db_session.execute(
            text(
                "INSERT INTO ai_section_extractions "
                "(run_item_id, document_id, section, data, prompt_version, schema_version) "
                "VALUES (:i, :d, :s, CAST(:data AS JSONB), 'sec-1', 'sec-1')"
            ),
            {
                "i": item_id,
                "d": document_id,
                "s": section,
                "data": json.dumps({"section": section, "fields": fields, "flags": flags or []}),
            },
        )
        db_session.commit()

    return _seed


def _item(db_session, item_id):
    return (
        db_session.execute(
            text(
                "SELECT section_row_id, filed_section, source_key "
                "FROM ai_processing_run_items WHERE id = :id"
            ),
            {"id": item_id},
        )
        .mappings()
        .one()
    )


def _reports_count(db_session, seed) -> int:
    return int(
        db_session.execute(
            text("SELECT count(*) FROM reports WHERE filepath = :k"),
            {"k": "reports/" + seed.source_key.split("/", 1)[1]},
        ).scalar_one()
    )


def _file(db_session, s3_client, bucket, seed, section, **overrides):
    kwargs = {
        "item_id": seed.item_id,
        "document_id": seed.document_id,
        "section": section,
        "content": build_content(db_session, seed.item_id, state=ContentState.CLASSIFIED),
        "bucket": bucket,
        "expected": {"classifying"},
    }
    kwargs.update(overrides)
    return filing.file_document(db_session, s3_client, **kwargs)


def test_filing_creates_the_section_row_and_moves_the_object(
    db_session, s3_client, bucket, classified_item
) -> None:
    seed = classified_item(DocumentSection.VACCINATIONS)
    content = build_content(db_session, seed.item_id, state=ContentState.CLASSIFIED)

    row_id = filing.file_document(
        db_session,
        s3_client,
        item_id=seed.item_id,
        document_id=seed.document_id,
        section=DocumentSection.VACCINATIONS,
        content=content,
        bucket=bucket,
        expected={"classifying"},
    )

    assert row_id is not None
    row = db_session.execute(select(vaccinations).where(vaccinations.c.id == row_id)).one()
    assert row.filepath.startswith("vaccinations/")
    assert row.content["ai"]["state"] == "classified"
    # Intake row gone, object relocated, original removed.
    assert (
        db_session.execute(
            select(unclassified_files.c.id).where(unclassified_files.c.id == seed.document_id)
        ).one_or_none()
        is None
    )
    assert object_exists(s3_client, bucket, row.filepath) is True
    assert object_exists(s3_client, bucket, seed.source_key) is False

    item = _item(db_session, seed.item_id)
    assert item["section_row_id"] == row_id
    assert item["filed_section"] == "vaccinations"
    # Later stages must find the document at its NEW key.
    assert item["source_key"] == row.filepath


def _set_intake_name(db_session, document_id: int, name: str) -> None:
    db_session.execute(
        text("UPDATE unclassified_files SET name = :n WHERE id = :d"),
        {"n": name, "d": document_id},
    )
    db_session.commit()


def _set_document_date(db_session, item_id, value: str) -> None:
    db_session.execute(
        text(
            "UPDATE ai_report_classifications SET document_date = CAST(:v AS date), "
            "document_date_label = 'Sample Collected' WHERE run_item_id = :i"
        ),
        {"v": value, "i": item_id},
    )
    db_session.commit()


def _filed_row(db_session, table, row_id):
    return db_session.execute(select(table.c.name, table.c.date).where(table.c.id == row_id)).one()


def test_filing_carries_the_users_filename_and_the_documents_date(
    db_session, s3_client, bucket, classified_item
):
    # Both were lost before these columns existed: the filename was dropped by every
    # mover, and the date was never read at all, so the app fell back to created_at --
    # the moment the AI filed the row, which for a 2019 report is today.
    seed = classified_item(DocumentSection.REPORTS)
    _set_intake_name(db_session, seed.document_id, "March bloods.pdf")
    _set_document_date(db_session, seed.item_id, "2026-03-12")

    row_id = _file(db_session, s3_client, bucket, seed, DocumentSection.REPORTS)

    row = _filed_row(db_session, reports, row_id)
    assert row.name == "March bloods.pdf"
    assert row.date == datetime(2026, 3, 12, tzinfo=UTC)


def test_the_date_is_midnight_utc_not_midnight_wherever_the_worker_runs(
    db_session, s3_client, bucket, classified_item
):
    # The column is timestamptz and the value we have is a date. A NAIVE midnight is sent
    # without a zone and Postgres reads it in the session's, so a worker running anywhere
    # but UTC would store a different instant and the app would show the wrong day. The
    # session timezone is moved here deliberately: with it left at UTC this test passes
    # whether or not the code attaches a zone, and proves nothing.
    seed = classified_item(DocumentSection.REPORTS)
    _set_document_date(db_session, seed.item_id, "2026-03-12")
    db_session.execute(text("SET LOCAL TIME ZONE 'America/New_York'"))

    row_id = _file(db_session, s3_client, bucket, seed, DocumentSection.REPORTS)

    assert _filed_row(db_session, reports, row_id).date == datetime(2026, 3, 12, tzinfo=UTC)


def test_filing_falls_back_to_the_ai_title_when_the_upload_was_unnamed(
    db_session, s3_client, bucket, classified_item
):
    # A global upload often carries no filename. Without the fallback the list reads
    # "Lab Report on 4 Aug 2026" for every one of them.
    seed = classified_item(DocumentSection.REPORTS)

    row_id = _file(db_session, s3_client, bucket, seed, DocumentSection.REPORTS)

    assert _filed_row(db_session, reports, row_id).name == "Seed Document"


def test_the_users_filename_beats_the_ai_title(db_session, s3_client, bucket, classified_item):
    seed = classified_item(DocumentSection.REPORTS)
    _set_intake_name(db_session, seed.document_id, "March bloods.pdf")

    row_id = _file(db_session, s3_client, bucket, seed, DocumentSection.REPORTS)

    assert _filed_row(db_session, reports, row_id).name == "March bloods.pdf"


def test_a_whitespace_filename_falls_back_rather_than_filing_a_blank_name(
    db_session, s3_client, bucket, classified_item
):
    seed = classified_item(DocumentSection.REPORTS)
    _set_intake_name(db_session, seed.document_id, "   ")

    row_id = _file(db_session, s3_client, bucket, seed, DocumentSection.REPORTS)

    assert _filed_row(db_session, reports, row_id).name == "Seed Document"


def test_a_document_with_no_readable_date_files_with_a_null_one(
    db_session, s3_client, bucket, classified_item
):
    seed = classified_item(DocumentSection.VACCINATIONS)

    row_id = _file(db_session, s3_client, bucket, seed, DocumentSection.VACCINATIONS)

    assert _filed_row(db_session, vaccinations, row_id).date is None


def test_filing_moves_the_preview_when_there_is_one(
    db_session, s3_client, bucket, classified_item
) -> None:
    seed = classified_item(DocumentSection.REPORTS, with_preview=True)

    row_id = filing.file_document(
        db_session,
        s3_client,
        item_id=seed.item_id,
        document_id=seed.document_id,
        section=DocumentSection.REPORTS,
        content=build_content(db_session, seed.item_id, state=ContentState.CLASSIFIED),
        bucket=bucket,
        expected={"classifying"},
    )

    assert row_id is not None
    new_key = _item(db_session, seed.item_id)["source_key"]
    assert new_key.startswith("reports/")
    assert object_exists(s3_client, bucket, preview_key_for(new_key)) is True
    # The old preview went with the original.
    assert object_exists(s3_client, bucket, preview_key_for(seed.source_key)) is False


def test_filing_without_a_preview_files_the_document_anyway(
    db_session, s3_client, bucket, classified_item
) -> None:
    """A preview is optional: most uploads have none and filing must not depend on it."""
    seed = classified_item(DocumentSection.INSURANCE)

    row_id = filing.file_document(
        db_session,
        s3_client,
        item_id=seed.item_id,
        document_id=seed.document_id,
        section=DocumentSection.INSURANCE,
        content=build_content(db_session, seed.item_id, state=ContentState.CLASSIFIED),
        bucket=bucket,
        expected={"classifying"},
    )

    assert row_id is not None
    assert _item(db_session, seed.item_id)["filed_section"] == "insurance"


def test_filing_twice_creates_only_one_row(db_session, s3_client, bucket, classified_item) -> None:
    """SQS is at-least-once. A redelivered filing must return the existing row."""
    seed = classified_item(DocumentSection.REPORTS)
    kwargs = {
        "item_id": seed.item_id,
        "document_id": seed.document_id,
        "section": DocumentSection.REPORTS,
        "content": build_content(db_session, seed.item_id, state=ContentState.CLASSIFIED),
        "bucket": bucket,
        "expected": {"classifying"},
    }

    first = filing.file_document(db_session, s3_client, **kwargs)
    second = filing.file_document(db_session, s3_client, **kwargs)

    assert first is not None
    assert first == second
    assert (
        db_session.execute(
            text("SELECT count(*) FROM reports WHERE filepath = :k"),
            {"k": _item(db_session, seed.item_id)["source_key"]},
        ).scalar_one()
        == 1
    )


def test_filing_a_vanished_document_is_a_permanent_reject(
    db_session, s3_client, bucket, classified_item
) -> None:
    """Something else filed it. Never fabricate a section row."""
    seed = classified_item(DocumentSection.REPORTS)
    db_session.execute(
        unclassified_files.delete().where(unclassified_files.c.id == seed.document_id)
    )
    db_session.commit()

    with pytest.raises(RejectStageError) as exc:
        filing.file_document(
            db_session,
            s3_client,
            item_id=seed.item_id,
            document_id=seed.document_id,
            section=DocumentSection.REPORTS,
            content={},
            bucket=bucket,
            expected={"classifying"},
        )

    assert exc.value.code == "source_document_missing"
    assert _item(db_session, seed.item_id)["section_row_id"] is None


def test_an_earlier_unfiled_attempt_is_not_mistaken_for_a_prior_filing(
    db_session, s3_client, bucket, classified_item
) -> None:
    """Adoption looks for a prior item that actually *filed* something.

    A document can have failed before filing and been retried, so an earlier item exists
    with a null `section_row_id`. Matching that one would report `section_changed_on_retry`
    (its `filed_section` is null) instead of the truth — that nothing was ever filed.
    """
    seed = classified_item(DocumentSection.REPORTS)
    db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status, source_key) "
            "SELECT run_id, document_id, 'failed', source_key FROM ai_processing_run_items "
            "WHERE id = :id"
        ),
        {"id": seed.item_id},
    )
    db_session.execute(
        unclassified_files.delete().where(unclassified_files.c.id == seed.document_id)
    )
    db_session.commit()

    with pytest.raises(RejectStageError) as exc:
        filing.file_document(
            db_session,
            s3_client,
            item_id=seed.item_id,
            document_id=seed.document_id,
            section=DocumentSection.REPORTS,
            content={},
            bucket=bucket,
            expected={"classifying"},
        )

    assert exc.value.code == "source_document_missing"


def test_a_cancel_during_filing_wins_and_files_nothing(
    db_session, s3_client, bucket, classified_item
) -> None:
    seed = classified_item(DocumentSection.REPORTS, with_preview=True)

    row_id = filing.file_document(
        db_session,
        s3_client,
        item_id=seed.item_id,
        document_id=seed.document_id,
        section=DocumentSection.REPORTS,
        content=build_content(db_session, seed.item_id, state=ContentState.CLASSIFIED),
        bucket=bucket,
        # Guard that cannot match: stands in for the item having been cancelled.
        expected={"generating_insights"},
    )

    assert row_id is None
    # The whole transaction rolled back: no orphan row, intake row intact.
    assert (
        db_session.execute(
            select(unclassified_files.c.id).where(unclassified_files.c.id == seed.document_id)
        ).one_or_none()
        is not None
    )
    orphans = db_session.execute(
        text("SELECT count(*) FROM reports WHERE filepath LIKE :k"),
        {"k": "reports/" + seed.source_key.split("/", 1)[1]},
    ).scalar_one()
    assert orphans == 0
    # THE invariant the post-commit delete ordering exists for: the intake row still points
    # at a live object. A pre-commit delete would be indistinguishable on the happy path.
    assert object_exists(s3_client, bucket, seed.source_key) is True
    assert object_exists(s3_client, bucket, preview_key_for(seed.source_key)) is True


def test_write_content_updates_the_filed_row(db_session, s3_client, bucket, classified_item):
    seed = classified_item(DocumentSection.REPORTS)
    row_id = filing.file_document(
        db_session,
        s3_client,
        item_id=seed.item_id,
        document_id=seed.document_id,
        section=DocumentSection.REPORTS,
        content=build_content(db_session, seed.item_id, state=ContentState.CLASSIFIED),
        bucket=bucket,
        expected={"classifying"},
    )

    assert (
        filing.write_content(
            db_session,
            seed.item_id,
            build_content(db_session, seed.item_id, state=ContentState.COMPLETE),
        )
        is True
    )

    content = db_session.execute(
        select(reports.c.content).where(reports.c.id == row_id)
    ).scalar_one()
    assert content["ai"]["state"] == "complete"


def test_mark_content_failed_stamps_the_filed_row(
    db_session, s3_client, bucket, classified_item
) -> None:
    """Otherwise a document that never finishes shows as 'processing' for ever."""
    seed = classified_item(DocumentSection.REPORTS)
    row_id = filing.file_document(
        db_session,
        s3_client,
        item_id=seed.item_id,
        document_id=seed.document_id,
        section=DocumentSection.REPORTS,
        content=build_content(db_session, seed.item_id, state=ContentState.CLASSIFIED),
        bucket=bucket,
        expected={"classifying"},
    )

    filing.mark_content_failed(db_session, seed.item_id)

    content = db_session.execute(
        select(reports.c.content).where(reports.c.id == row_id)
    ).scalar_one()
    assert content["ai"]["state"] == "failed"


def test_write_content_on_an_unfiled_item_is_a_no_op(db_session, classified_item) -> None:
    """A mismatch or an unsupported section is never filed, so there is nothing to update."""
    seed = classified_item(DocumentSection.REPORTS)
    assert filing.write_content(db_session, seed.item_id, {"ai": {}}) is False


def test_vaccination_next_due_on_comes_from_the_extraction(
    db_session, s3_client, bucket, classified_item, seed_section_extraction
) -> None:
    seed = classified_item(DocumentSection.VACCINATIONS)
    seed_section_extraction(
        seed.item_id, seed.document_id, "vaccinations", {"next_due_date": "2027-03-14"}
    )
    row_id = filing.file_document(
        db_session,
        s3_client,
        item_id=seed.item_id,
        document_id=seed.document_id,
        section=DocumentSection.VACCINATIONS,
        content=build_content(db_session, seed.item_id, state=ContentState.CLASSIFIED),
        bucket=bucket,
        expected={"classifying"},
    )

    filing.write_content(
        db_session,
        seed.item_id,
        build_content(db_session, seed.item_id, state=ContentState.COMPLETE),
        extra=filing.extra_columns(db_session, seed.item_id, DocumentSection.VACCINATIONS),
    )

    due = db_session.execute(
        select(vaccinations.c.next_due_on).where(vaccinations.c.id == row_id)
    ).scalar_one()
    assert due is not None
    assert (due.year, due.month, due.day) == (2027, 3, 14)


def test_a_reminder_is_not_set_from_dates_we_flagged_as_inconsistent(
    db_session, classified_item, seed_section_extraction
) -> None:
    """`dates_out_of_order` means the next dose reads as EARLIER than the dose given.

    That is a misread, and `_date_flags` exists to keep such values "visible for a human
    without asserting they are correct". This is the one consumer that ACTS on the value
    — `vaccinations.next_due_on` drives Spring's reminder index — so it is the one place
    that assertion would have been made. The content still carries both dates; only the
    reminder is withheld.
    """
    seed = classified_item(DocumentSection.VACCINATIONS)
    seed_section_extraction(
        seed.item_id,
        seed.document_id,
        "vaccinations",
        {"date_given": "2027-03-14", "next_due_date": "2026-01-01"},
        flags=[{"code": "dates_out_of_order", "field": "next_due_date", "detail": "..."}],
    )

    assert filing.extra_columns(db_session, seed.item_id, DocumentSection.VACCINATIONS) == {}


def test_extra_columns_is_empty_for_a_section_with_none(db_session, classified_item) -> None:
    seed = classified_item(DocumentSection.REPORTS)
    assert filing.extra_columns(db_session, seed.item_id, DocumentSection.REPORTS) == {}


def test_bill_amounts_reach_the_columns_the_app_reads(
    db_session, s3_client, bucket, classified_item, seed_section_extraction
) -> None:
    """`bills` has columns for the very fields we extract, so the extraction fills them.

    Leaving them null would mean the bills list renders no amount while the value sits in
    `content` — which is the whole point of extracting it.
    """
    seed = classified_item(DocumentSection.BILLS)
    seed_section_extraction(
        seed.item_id,
        seed.document_id,
        "bills",
        # As `build_payload` stores them: bare decimal strings and an ISO currency code.
        {"total_amount": "1450.00", "amount_due": "450.50", "currency": "INR"},
    )
    row_id = filing.file_document(
        db_session,
        s3_client,
        item_id=seed.item_id,
        document_id=seed.document_id,
        section=DocumentSection.BILLS,
        content=build_content(db_session, seed.item_id, state=ContentState.CLASSIFIED),
        bucket=bucket,
        expected={"classifying"},
    )

    filing.write_content(
        db_session,
        seed.item_id,
        build_content(db_session, seed.item_id, state=ContentState.COMPLETE),
        extra=filing.extra_columns(db_session, seed.item_id, DocumentSection.BILLS),
    )

    row = (
        db_session.execute(
            select(bills.c.amount, bills.c.amount_due, bills.c.amount_currency).where(
                bills.c.id == row_id
            )
        )
        .mappings()
        .one()
    )
    assert str(row["amount"]) == "1450.00"
    assert str(row["amount_due"]) == "450.50"
    assert row["amount_currency"] == "INR"


def test_a_bill_amount_the_column_cannot_hold_is_skipped_not_coerced(
    db_session, classified_item, seed_section_extraction
) -> None:
    """`amount` is numeric(10, 2) and `amount_currency` is a four-value enum.

    A misread that ran two columns together, or a currency outside the enum, would fail the
    UPDATE — losing the `content` write for a document that was otherwise processed fine.
    Each value is checked against its column's shape on its own, so the readable ones still
    land and the unreadable ones stay in `content` for a human.
    """
    seed = classified_item(DocumentSection.BILLS)
    seed_section_extraction(
        seed.item_id,
        seed.document_id,
        "bills",
        # `normalise_currency` passes any three letters through, so a legal ISO code with no
        # place in the enum is the realistic case — not a garbled one.
        {"total_amount": "300000500000", "amount_due": "450.50", "currency": "AED"},
    )

    assert filing.extra_columns(db_session, seed.item_id, DocumentSection.BILLS) == {
        "amount_due": Decimal("450.50")
    }


# --- concurrency: one document, one section row -----------------------------


def test_the_already_filed_check_takes_a_row_lock(
    db_connection, db_session, s3_client, bucket, classified_item
) -> None:
    """Without FOR UPDATE the guarded UPDATE is not self-excluding (it leaves `status`
    alone), so two workers on one document would both file it. Pins the lock itself,
    because the interleaving it prevents needs two connections to reproduce."""
    statements: list[str] = []

    @event.listens_for(db_connection, "before_cursor_execute")
    def _capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    try:
        seed = classified_item(DocumentSection.REPORTS)
        statements.clear()
        _file(db_session, s3_client, bucket, seed, DocumentSection.REPORTS)
    finally:
        event.remove(db_connection, "before_cursor_execute", _capture)

    locking = [s for s in statements if "FOR UPDATE" in s and "ai_processing_run_items" in s]
    assert locking, "the already-filed check must SELECT ... FOR UPDATE"


def test_a_rival_filing_recorded_first_is_never_overwritten(
    db_session, s3_client, bucket, classified_item, monkeypatch
) -> None:
    """The UPDATE guard's `section_row_id IS NULL` half. Stands in for a second worker
    having filed and committed while we were copying: we must file nothing."""
    seed = classified_item(DocumentSection.REPORTS)

    def _rival_files_it(*_args, **_kwargs) -> bool:
        # Runs after our SELECT and before the guarded UPDATE.
        db_session.execute(
            text("UPDATE ai_processing_run_items SET section_row_id = 999999 WHERE id = :i"),
            {"i": seed.item_id},
        )
        return False

    monkeypatch.setattr(filing, "object_exists", _rival_files_it)

    row_id = _file(db_session, s3_client, bucket, seed, DocumentSection.REPORTS)

    assert row_id is None
    assert _reports_count(db_session, seed) == 0


# --- S3 failures obey the stage error contract ------------------------------


def test_a_vanished_object_during_filing_is_a_permanent_reject(
    db_session, s3_client, bucket, classified_item, monkeypatch
) -> None:
    """A user hand-filing in Spring mid-copy. Retrying cannot bring it back, and each
    retry re-pays for classification, so it must be terminal — not a bare exception."""
    seed = classified_item(DocumentSection.REPORTS)

    def _gone(*_args, **_kwargs) -> None:
        raise SourceObjectMissingError("unclassified/secret-key.pdf")

    monkeypatch.setattr(filing, "copy_object", _gone)

    with pytest.raises(RejectStageError) as exc:
        _file(db_session, s3_client, bucket, seed, DocumentSection.REPORTS)

    assert exc.value.code == "source_object_missing"
    # Keys are internal detail and never travel in an error body.
    assert "secret-key" not in exc.value.message
    assert _reports_count(db_session, seed) == 0


def test_s3_being_unreachable_during_filing_is_transient(
    db_session, s3_client, bucket, classified_item, monkeypatch
) -> None:
    seed = classified_item(DocumentSection.REPORTS)

    def _blip(*_args, **_kwargs) -> None:
        raise SourceObjectUnavailableError("SlowDown")

    monkeypatch.setattr(filing, "copy_object", _blip)

    with pytest.raises(TransientStageError):
        _file(db_session, s3_client, bucket, seed, DocumentSection.REPORTS)

    assert _reports_count(db_session, seed) == 0
