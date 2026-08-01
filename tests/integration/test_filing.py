"""Filing a classified document into its section table.

The ordering under test is the point: S3 copy, then one DB transaction, then the S3 delete.
Any other order can leave a live row pointing at a deleted object.
"""

import json
import uuid
from typing import NamedTuple

import pytest
from sqlalchemy import event, select, text

from app.integrations.s3 import (
    SourceObjectMissingError,
    SourceObjectUnavailableError,
    object_exists,
)
from app.models.spring import reports, unclassified_files, vaccinations
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
    def _seed(item_id, document_id, section: str, fields: dict) -> None:
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
                "data": json.dumps({"section": section, "fields": fields, "flags": []}),
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


def test_extra_columns_is_empty_for_a_non_vaccination_section(db_session, classified_item) -> None:
    seed = classified_item(DocumentSection.REPORTS)
    assert filing.extra_columns(db_session, seed.item_id, DocumentSection.REPORTS) == {}


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
