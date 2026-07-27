"""End-to-end: the extraction stage applies approved-THP age-group ideal ranges over the
report's own range, logs an R&D worklist on fallback, and is a no-op when the flag is off.

The Spring parameter tables do not exist in the base schema, so this test creates minimal
stand-ins inside its own rolled-back transaction. Keep their shape in sync with the
bindings in app/models/spring.py and the confirmed Spring contract.
"""

import uuid

import pytest
from sqlalchemy import text

from app.services.extraction import extract_report
from app.workers.stagetypes import StageContext
from tests.support.ai import FakeAIProvider, extraction_payload, structured_response

pytestmark = pytest.mark.integration


@pytest.fixture
def thp_tables(db_session):
    """Create + seed minimal parameter tables in the test's transaction (rolled back)."""
    db_session.execute(
        text(
            "CREATE TABLE parameters (pkid bigint PRIMARY KEY, name varchar(155), "
            "status varchar(50), approved_by_id integer)"
        )
    )
    db_session.execute(
        text(
            "CREATE TABLE parameter_aliases (id serial PRIMARY KEY, parameter_id bigint, "
            "alias varchar(255))"
        )
    )
    db_session.execute(
        text(
            "CREATE TABLE parameter_ideal_values (id serial PRIMARY KEY, "
            'parameter_id bigint, "group" varchar(50), ideal_value_min double precision, '
            "ideal_value_max double precision)"
        )
    )
    db_session.execute(
        text(
            "INSERT INTO parameters (pkid, name, status, approved_by_id) VALUES "
            "(1, 'Fasting Glucose', 'approved', 7), "  # approved
            "(2, 'Cholesterol', 'pending', NULL)"  # not approved
        )
    )
    db_session.execute(
        text("INSERT INTO parameter_aliases (parameter_id, alias) VALUES (1, 'Glucose Fasting')")
    )
    db_session.execute(
        text(
            'INSERT INTO parameter_ideal_values (parameter_id, "group", '
            "ideal_value_min, ideal_value_max) VALUES (1, 'Adult Male', 70, 90)"
        )
    )
    db_session.flush()


def _seed_item(db_session, document_id: int) -> uuid.UUID:
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, document_id, status) "
            "VALUES (:r, :rep, 'extracting') RETURNING id"
        ),
        {"r": run_id, "rep": document_id},
    ).scalar_one()
    db_session.flush()
    return item_id


def _context(db_session, aws, settings, document_id, item_id, ai) -> StageContext:
    return StageContext(
        item_id=item_id,
        run_id=uuid.uuid4(),
        document_id=document_id,
        attempt=1,
        session=db_session,
        s3=aws[0],
        ai=ai,
        settings=settings,
    )


def _enabled(settings):
    return settings.model_copy(update={"ideal_ranges_enabled": True})


def _payload(test_name: str, value: str, reference_range: str):
    return extraction_payload(
        results=[
            {
                "test_name": test_name,
                "value": value,
                "unit": "mg/dL",
                "reference_range": reference_range,
                "observed_date": None,
                "source_context": None,
            }
        ],
        patient_age="45",
        patient_gender="Male",
    )


def _result(db_session, item_id):
    return db_session.execute(
        text("SELECT data FROM ai_report_extractions WHERE run_item_id = :id"),
        {"id": item_id},
    ).scalar_one()["results"][0]


def _fallbacks(db_session, item_id):
    return (
        db_session.execute(
            text("SELECT * FROM ai_thp_fallbacks WHERE run_item_id = :id"),
            {"id": item_id},
        )
        .mappings()
        .all()
    )


def test_approved_ideal_range_overrides_report_range(
    db_session, make_document, aws, test_settings, thp_tables
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    # Report range 70-99 says 95 is normal; the approved ideal range 70-90 says high.
    ai = FakeAIProvider(response=structured_response(_payload("Fasting Glucose", "95", "70-99")))

    extract_report(_context(db_session, aws, _enabled(test_settings), document_id, item_id, ai))

    result = _result(db_session, item_id)
    assert result["range_source"] == "ideal_range"
    assert result["abnormal_flag"] == "high"  # driven by the ideal range, not 70-99
    assert result["matched_parameter"] == "Fasting Glucose"
    assert result["matched_group"] == "adult male"
    assert _fallbacks(db_session, item_id) == []


def test_unmatched_test_falls_back_and_logs_worklist(
    db_session, make_document, aws, test_settings, thp_tables
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider(response=structured_response(_payload("Obscure Marker Z", "5", "1-10")))

    extract_report(_context(db_session, aws, _enabled(test_settings), document_id, item_id, ai))

    result = _result(db_session, item_id)
    assert result["range_source"] == "report_range"
    assert result["abnormal_flag"] == "normal"  # 5 in 1-10 from the report
    rows = _fallbacks(db_session, item_id)
    assert len(rows) == 1
    assert rows[0]["reason"] == "unmatched"
    assert rows[0]["test_name"] == "Obscure Marker Z"


def test_unapproved_parameter_falls_back(db_session, make_document, aws, test_settings, thp_tables):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider(response=structured_response(_payload("Cholesterol", "180", "0-200")))

    extract_report(_context(db_session, aws, _enabled(test_settings), document_id, item_id, ai))

    assert _result(db_session, item_id)["range_source"] == "report_range"
    rows = _fallbacks(db_session, item_id)
    assert len(rows) == 1
    assert rows[0]["reason"] == "unapproved"
    assert rows[0]["matched_parameter"] == "Cholesterol"


def test_redelivery_does_not_duplicate_fallbacks(
    db_session, make_document, aws, test_settings, thp_tables
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider(response=structured_response(_payload("Obscure Marker Z", "5", "1-10")))
    ctx = _context(db_session, aws, _enabled(test_settings), document_id, item_id, ai)

    extract_report(ctx)
    extract_report(ctx)  # redelivery re-runs the same attempt

    assert len(_fallbacks(db_session, item_id)) == 1


def test_flag_off_is_unchanged_and_writes_no_worklist(
    db_session, make_document, aws, test_settings, thp_tables
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    # Same value/range as the override test, but the flag is OFF -> report range wins.
    ai = FakeAIProvider(response=structured_response(_payload("Fasting Glucose", "95", "70-99")))

    extract_report(_context(db_session, aws, test_settings, document_id, item_id, ai))

    result = _result(db_session, item_id)
    assert result["range_source"] == "report_range"
    assert result["abnormal_flag"] == "normal"  # 95 in 70-99
    assert _fallbacks(db_session, item_id) == []
