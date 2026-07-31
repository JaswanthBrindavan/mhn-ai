"""End-to-end: the extraction stage applies approved-THP age-bracket ideal ranges over the
report's own range, logs an R&D worklist on fallback, and is a no-op when the flag is off.

The THP tables are Spring-owned base schema, so this seeds real rows inside the test's own
rolled-back transaction rather than creating stand-ins. Ids come back from ``RETURNING``
instead of being hardcoded, so the fixture cannot collide with curated data.
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
    """Seed one approved and one unapproved THP in the test's transaction (rolled back)."""
    approved_id = db_session.execute(
        text(
            "INSERT INTO traditional_health_parameters (name, units, approved, visible, aliases) "
            "VALUES ('Fasting Glucose', 'mg/dL', true, true, ARRAY['Glucose Fasting']) "
            "RETURNING id"
        )
    ).scalar_one()
    db_session.execute(
        text(
            "INSERT INTO traditional_health_parameters (name, units, approved, visible, aliases) "
            "VALUES ('Cholesterol', 'mg/dL', false, true, NULL)"  # not doctor-approved
        )
    )
    # Ideal 70-90 for adults. low_warn/high_warn are the pair we flag against; the danger
    # and graph bounds are seeded much wider to prove they are not what drives the flag.
    db_session.execute(
        text(
            "INSERT INTO thp_age_range (thp_id, age_min, age_max, min, low_danger, low_warn, "
            "ideal, high_warn, high_danger, max) VALUES (:t, 18, 60, 0, 50, 70, 80, 90, 200, 500)"
        ),
        {"t": approved_id},
    )
    db_session.execute(
        text(
            "INSERT INTO thp_alternate_units (thp_id, name, multiplier, offset_value) "
            "VALUES (:t, 'mmol/L', 18.0, 0)"  # mg/dL = mmol/L * 18
        ),
        {"t": approved_id},
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


def _payload(test_name: str, value: str, reference_range: str, unit: str = "mg/dL"):
    return extraction_payload(
        results=[
            {
                "test_name": test_name,
                "value": value,
                "unit": unit,
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
    assert result["matched_group"] == "18-60"
    assert _fallbacks(db_session, item_id) == []


def test_alias_and_alternate_unit_are_converted(
    db_session, make_document, aws, test_settings, thp_tables
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    # Matched by alias, printed in mmol/L. The ideal range 70-90 mg/dL becomes 3.89-5.0
    # mmol/L, so 5.5 is high — while the report's own range would have called it normal.
    ai = FakeAIProvider(
        response=structured_response(_payload("Glucose Fasting", "5.5", "3.0-6.0", unit="mmol/L"))
    )

    extract_report(_context(db_session, aws, _enabled(test_settings), document_id, item_id, ai))

    result = _result(db_session, item_id)
    assert result["range_source"] == "ideal_range"
    assert result["abnormal_flag"] == "high"
    assert result["matched_parameter"] == "Fasting Glucose"
    assert _fallbacks(db_session, item_id) == []


def test_uncurated_unit_falls_back_and_logs_the_unit(
    db_session, make_document, aws, test_settings, thp_tables
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    # g/L has no curated conversion, so the mg/dL ideal range must not be compared to it.
    ai = FakeAIProvider(
        response=structured_response(_payload("Fasting Glucose", "0.95", "0.7-1.1", unit="g/L"))
    )

    extract_report(_context(db_session, aws, _enabled(test_settings), document_id, item_id, ai))

    result = _result(db_session, item_id)
    assert result["range_source"] == "report_range"
    assert result["abnormal_flag"] == "normal"  # 0.95 in the report's own 0.7-1.1
    rows = _fallbacks(db_session, item_id)
    assert len(rows) == 1
    assert rows[0]["reason"] == "unit_mismatch"
    assert rows[0]["report_unit"] == "g/L"
    assert rows[0]["group_attempted"] == "18-60"  # the bracket resolved; only the unit failed


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
