"""End-to-end: the extraction stage applies approved-THP age-bracket ideal ranges over the
report's own range, logs an R&D worklist on fallback, and is a no-op when the flag is off.

The THP tables are Spring-owned base schema, so this seeds real rows inside the test's own
rolled-back transaction rather than creating stand-ins. Ids come back from ``RETURNING``
instead of being hardcoded, so the fixture cannot collide with curated data.

**Every seeded name carries a [fixture] marker**, because since Spring's V18 the database
holds the real catalogue — 193 parameters and 1184 aliases — and BOTH
``traditional_health_parameters.name`` and ``thp_alias.alias`` are globally unique. A plain
'Ferritin' collides with the curated row rather than shadowing it.
"""

import uuid

import pytest
from sqlalchemy import text

from app.services.extraction import extract_report
from app.workers.stagetypes import StageContext
from tests.integration.conftest import document_key
from tests.support.ai import FakeAIProvider, extraction_payload, structured_response

pytestmark = pytest.mark.integration

GLUCOSE = "Fasting Glucose [fixture]"
GLUCOSE_ALIAS = "Glucose Fasting [fixture]"
GLUCOSE_PENDING_ALIAS = "Sugar F [fixture]"
CHOLESTEROL = "Cholesterol [fixture]"
CREATININE = "Creatinine [fixture]"
FERRITIN = "Ferritin [fixture]"


def _thp(db_session, name: str, *, status: str = "approved", ai: bool = True) -> int:
    """One parameter in the master, returning its id.

    ``status`` is the doctor-approval gate (Spring's reference_status_enum). The legacy
    ``approved`` boolean is left alone: V14's ``trg_thp_status_mirror`` derives it from
    ``status``, so setting it here would prove nothing about what this service reads.
    """
    return db_session.execute(
        text(
            "INSERT INTO traditional_health_parameters (name, units, status, ai_integrated, "
            "visible) VALUES (:n, 'mg/dL', :s, :ai, true) RETURNING id"
        ),
        {"n": name, "s": status, "ai": ai},
    ).scalar_one()


def _bracket(db_session, thp_id: int, low: float, high: float, *, sex: str = "any") -> None:
    """One curated ideal range. ``min``/``max`` are the graph bounds and are seeded much
    wider than the ideal band to prove they are not what drives the flag."""
    db_session.execute(
        text(
            "INSERT INTO thp_age_range (thp_id, sex, age_min, age_max, min, low_warn, ideal, "
            "high_warn, max) VALUES (:t, :sex, 18, 60, 0, :low, :mid, :high, 500)"
        ),
        {"t": thp_id, "sex": sex, "low": low, "high": high, "mid": (low + high) / 2},
    )


@pytest.fixture
def thp_tables(db_session):
    """Seed the curated catalogue this service reads, in the test's rolled-back transaction.

    Deliberately seeded through the CURRENT columns — ``status``, ``ai_integrated``,
    ``thp_alias``, ``sex`` — because the superseded ones (the ``approved`` boolean, the
    inline ``aliases`` array) still exist in the schema, so reading the wrong one returns
    nothing rather than erroring.
    """
    approved_id = _thp(db_session, GLUCOSE)
    db_session.execute(
        text("INSERT INTO thp_alias (thp_id, alias, status) VALUES (:t, :a, :s)"),
        {"t": approved_id, "a": GLUCOSE_ALIAS, "s": "approved"},
    )
    # An alias still awaiting a reviewer must not decide anyone's flag.
    db_session.execute(
        text("INSERT INTO thp_alias (thp_id, alias, status) VALUES (:t, :a, :s)"),
        {"t": approved_id, "a": GLUCOSE_PENDING_ALIAS, "s": "pending"},
    )
    _bracket(db_session, approved_id, 70, 90)  # ideal 70-90 for adults
    db_session.execute(
        text(
            "INSERT INTO thp_alternate_units (thp_id, name, multiplier, offset_value) "
            "VALUES (:t, 'mmol/L', 18.0, 0)"  # mg/dL = mmol/L * 18
        ),
        {"t": approved_id},
    )

    _thp(db_session, CHOLESTEROL, status="draft")  # curated but not yet approved
    # Approved, fully curated, but withheld from AI processing on the staff dashboard.
    _bracket(db_session, _thp(db_session, CREATININE, ai=False), 70, 90)

    # A parameter whose range is curated per sex and has no 'any' row -- the shape of the
    # 15 parameters in Spring's V18 catalogue that are curated only per sex.
    ferritin_id = _thp(db_session, FERRITIN)
    _bracket(db_session, ferritin_id, 30, 400, sex="male")
    _bracket(db_session, ferritin_id, 15, 150, sex="female")
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
        source_key=document_key(db_session, document_id),
        attempt=1,
        session=db_session,
        s3=aws[0],
        ai=ai,
        settings=settings,
    )


def _enabled(settings):
    return settings.model_copy(update={"ideal_ranges_enabled": True})


def _payload(
    test_name: str,
    value: str,
    reference_range: str,
    unit: str = "mg/dL",
    gender: str | None = "Male",
):
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
        patient_gender=gender,
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
    ai = FakeAIProvider(response=structured_response(_payload(GLUCOSE, "95", "70-99")))

    extract_report(_context(db_session, aws, _enabled(test_settings), document_id, item_id, ai))

    result = _result(db_session, item_id)
    assert result["range_source"] == "ideal_range"
    assert result["abnormal_flag"] == "high"  # driven by the ideal range, not 70-99
    assert result["matched_parameter"] == GLUCOSE
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
        response=structured_response(_payload(GLUCOSE_ALIAS, "5.5", "3.0-6.0", unit="mmol/L"))
    )

    extract_report(_context(db_session, aws, _enabled(test_settings), document_id, item_id, ai))

    result = _result(db_session, item_id)
    assert result["range_source"] == "ideal_range"
    assert result["abnormal_flag"] == "high"
    assert result["matched_parameter"] == GLUCOSE
    assert _fallbacks(db_session, item_id) == []


def test_uncurated_unit_falls_back_and_logs_the_unit(
    db_session, make_document, aws, test_settings, thp_tables
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    # g/L has no curated conversion, so the mg/dL ideal range must not be compared to it.
    ai = FakeAIProvider(
        response=structured_response(_payload(GLUCOSE, "0.95", "0.7-1.1", unit="g/L"))
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


def test_sex_specific_bracket_uses_the_reports_own_sex(
    db_session, make_document, aws, test_settings, thp_tables
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    # 200 sits inside the male range (30-400) and above the female one (15-150). The report
    # says female, so it must come back high -- reading the male row would call it normal,
    # which is the bug a sex-blind bracket pick produced.
    ai = FakeAIProvider(
        response=structured_response(_payload(FERRITIN, "200", "20-300", gender="Female"))
    )

    extract_report(_context(db_session, aws, _enabled(test_settings), document_id, item_id, ai))

    result = _result(db_session, item_id)
    assert result["range_source"] == "ideal_range"
    assert result["abnormal_flag"] == "high"
    assert result["matched_group"] == "18-60 female"
    assert _fallbacks(db_session, item_id) == []


def test_sex_specific_bracket_is_refused_when_the_report_gives_no_sex(
    db_session, make_document, aws, test_settings, thp_tables
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider(
        response=structured_response(_payload(FERRITIN, "200", "20-300", gender=None))
    )

    extract_report(_context(db_session, aws, _enabled(test_settings), document_id, item_id, ai))

    result = _result(db_session, item_id)
    assert result["range_source"] == "report_range"  # never one half of a sex split
    assert result["abnormal_flag"] == "normal"  # 200 in the report's own 20-300
    rows = _fallbacks(db_session, item_id)
    assert len(rows) == 1
    assert rows[0]["reason"] == "no_ideal_range"
    assert rows[0]["matched_parameter"] == FERRITIN


def test_unapproved_alias_does_not_match(db_session, make_document, aws, test_settings, thp_tables):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    # 'Sugar (F)' is a pending alias for Fasting Glucose, so the test reads as unknown --
    # not as a matched-but-unapproved parameter.
    ai = FakeAIProvider(
        response=structured_response(_payload(GLUCOSE_PENDING_ALIAS, "95", "70-99"))
    )

    extract_report(_context(db_session, aws, _enabled(test_settings), document_id, item_id, ai))

    assert _result(db_session, item_id)["range_source"] == "report_range"
    rows = _fallbacks(db_session, item_id)
    assert len(rows) == 1
    assert rows[0]["reason"] == "unmatched"


def test_parameter_withheld_from_ai_falls_back(
    db_session, make_document, aws, test_settings, thp_tables
):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    # Approved and fully curated, but ai_integrated is false: the staff dashboard's own
    # switch for keeping a parameter out of AI processing.
    ai = FakeAIProvider(response=structured_response(_payload(CREATININE, "95", "70-99")))

    extract_report(_context(db_session, aws, _enabled(test_settings), document_id, item_id, ai))

    assert _result(db_session, item_id)["range_source"] == "report_range"
    rows = _fallbacks(db_session, item_id)
    assert len(rows) == 1
    assert rows[0]["reason"] == "unapproved"
    assert rows[0]["matched_parameter"] == CREATININE


def test_unapproved_parameter_falls_back(db_session, make_document, aws, test_settings, thp_tables):
    document_id = make_document()
    item_id = _seed_item(db_session, document_id)
    ai = FakeAIProvider(response=structured_response(_payload(CHOLESTEROL, "180", "0-200")))

    extract_report(_context(db_session, aws, _enabled(test_settings), document_id, item_id, ai))

    assert _result(db_session, item_id)["range_source"] == "report_range"
    rows = _fallbacks(db_session, item_id)
    assert len(rows) == 1
    assert rows[0]["reason"] == "unapproved"
    assert rows[0]["matched_parameter"] == CHOLESTEROL


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
    ai = FakeAIProvider(response=structured_response(_payload(GLUCOSE, "95", "70-99")))

    extract_report(_context(db_session, aws, test_settings, document_id, item_id, ai))

    result = _result(db_session, item_id)
    assert result["range_source"] == "report_range"
    assert result["abnormal_flag"] == "normal"  # 95 in 70-99
    assert _fallbacks(db_session, item_id) == []
