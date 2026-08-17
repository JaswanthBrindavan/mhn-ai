"""Accepting the classification: move a mismatched document, then process it there.

A document filed against our reading — the user put an insurance policy under Reports —
is filed but never read, and carries a `section_mismatch` flag. That flag is the
permission: it is the only state from which this action is offered, and its destination is
always the section we detected, because that is the only one we have an opinion about.

The move is the reverse mover `docs/auto-filing-design.md` said would never be built. What
keeps it narrow is the refusals below.
"""

import pytest
from sqlalchemy import select, text

from app.core.errors import ApiError
from app.integrations.s3 import object_exists
from app.models.spring import insurance, reports
from app.services import results as results_service

pytestmark = pytest.mark.integration


def _filed_mismatch(db_session, make_document, s3, bucket, *, detected="insurance"):
    """A document filed under `reports` while classified as `detected`, as the router
    leaves it — filed, unread, flagged."""
    document_id = make_document()
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    key = db_session.execute(
        text("SELECT filepath FROM unclassified_files WHERE id = :d"), {"d": document_id}
    ).scalar_one()
    reports_key = f"reports/{key.split('/', 1)[1]}"
    s3.copy_object(Bucket=bucket, CopySource=f"{bucket}/{key}", Key=reports_key)

    owner = db_session.execute(
        text("SELECT user_id FROM unclassified_files WHERE id = :d"), {"d": document_id}
    ).scalar_one()
    row_id = db_session.execute(
        reports.insert()
        .values(
            user_id=owner,
            filepath=reports_key,
            private=False,
            content={"ai": {"state": "complete"}},
        )
        .returning(reports.c.id)
    ).scalar_one()

    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items "
            "(run_id, document_id, status, intended_section, filed_section, section_row_id, "
            " source_key, last_error_code) "
            "VALUES (:r, :d, 'rejected', 'reports', 'reports', :row, :key, 'section_mismatch') "
            "RETURNING id"
        ),
        {"r": run_id, "d": document_id, "row": row_id, "key": reports_key},
    ).scalar_one()
    db_session.execute(
        text(
            "INSERT INTO ai_report_classifications "
            "(run_item_id, document_id, section, title, confidence, prompt_version, "
            " schema_version) VALUES (:i, :d, :s, 'A Policy', 0.94, 'clf-1', 'clf-1')"
        ),
        {"i": item_id, "d": document_id, "s": detected},
    )
    db_session.execute(text("DELETE FROM unclassified_files WHERE id = :d"), {"d": document_id})
    db_session.commit()
    return item_id, document_id, row_id, reports_key


def test_refile_moves_the_row_the_object_and_the_run_item(
    db_session, make_document, aws, test_settings
):
    s3, sqs, _, _ = aws
    bucket = test_settings.s3_bucket
    item_id, document_id, old_row_id, old_key = _filed_mismatch(
        db_session, make_document, s3, bucket
    )

    results_service.refile_document(
        db_session, document_id, None, s3=s3, sqs=sqs, settings=test_settings
    )

    item = (
        db_session.execute(
            text(
                "SELECT section_row_id, filed_section, source_key FROM ai_processing_run_items "
                "WHERE id = :i"
            ),
            {"i": item_id},
        )
        .mappings()
        .one()
    )
    assert item["filed_section"] == "insurance"
    assert item["section_row_id"] != old_row_id
    assert item["source_key"].startswith("insurance/")

    # The destination row exists and the source row is gone — one document, one row.
    assert (
        db_session.execute(
            select(insurance.c.filepath).where(insurance.c.id == item["section_row_id"])
        ).scalar_one()
        == item["source_key"]
    )
    assert (
        db_session.execute(select(reports.c.id).where(reports.c.id == old_row_id)).one_or_none()
        is None
    )

    # The object moved with it, and the original is gone.
    assert object_exists(s3, bucket, item["source_key"]) is True
    assert object_exists(s3, bucket, old_key) is False


def test_refile_submits_the_document_for_processing(db_session, make_document, aws, test_settings):
    """The move alone is half the point — the document has still never been read.

    A fresh run item is created and published; when the worker takes it,
    `_adopt_prior_filing` finds a prior filing whose section now MATCHES the detected one,
    so the stages update the moved row rather than filing a second copy.
    """
    s3, sqs, _, _ = aws
    item_id, document_id, _, _ = _filed_mismatch(
        db_session, make_document, s3, bucket=test_settings.s3_bucket
    )

    response = results_service.refile_document(
        db_session, document_id, None, s3=s3, sqs=sqs, settings=test_settings
    )

    assert response.item_id != item_id
    assert response.document_id == document_id


# --- what is NOT refilable --------------------------------------------------


def test_a_document_that_was_never_filed_is_refused(db_session, make_document, aws, test_settings):
    s3, sqs, _, _ = aws
    document_id = make_document()
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status) "
            "VALUES (:r, :d, 'rejected')"
        ),
        {"r": run_id, "d": document_id},
    )
    db_session.commit()

    with pytest.raises(ApiError) as exc:
        results_service.refile_document(
            db_session, document_id, None, s3=s3, sqs=sqs, settings=test_settings
        )
    assert exc.value.code == "not_filed"


def test_a_document_already_in_its_detected_section_is_refused(
    db_session, make_document, aws, test_settings
):
    """The action exists to resolve a disagreement. Without one there is nothing to do —
    and this is what stops a correctly filed document being shuffled around."""
    s3, sqs, _, _ = aws
    document_id = _filed_mismatch(
        db_session, make_document, s3, test_settings.s3_bucket, detected="reports"
    )[1]

    with pytest.raises(ApiError) as exc:
        results_service.refile_document(
            db_session, document_id, None, s3=s3, sqs=sqs, settings=test_settings
        )
    assert exc.value.code == "already_in_detected_section"


def test_a_detected_section_we_never_file_is_refused(db_session, make_document, aws, test_settings):
    """Classified `medical_condition` or `unknown`: there is nowhere to move it TO, because
    this service files neither. The app does not offer the action for them either."""
    s3, sqs, _, _ = aws
    document_id = _filed_mismatch(
        db_session, make_document, s3, test_settings.s3_bucket, detected="medical_condition"
    )[1]

    with pytest.raises(ApiError) as exc:
        results_service.refile_document(
            db_session, document_id, None, s3=s3, sqs=sqs, settings=test_settings
        )
    assert exc.value.code == "section_not_filable"


def test_an_unknown_document_id_is_a_404(db_session, aws, test_settings):
    s3, sqs, _, _ = aws
    with pytest.raises(ApiError) as exc:
        results_service.refile_document(
            db_session, 999_999, None, s3=s3, sqs=sqs, settings=test_settings
        )
    assert exc.value.status_code == 404
