import pytest

from app.services.ideal_ranges import (
    Lookup,
    in_report_unit,
    match_key,
    parse_age,
    pick_bracket,
    resolve,
)


def test_parse_age_units():
    assert parse_age("23") == 23.0
    assert parse_age("23 years") == 23.0
    assert parse_age("6 months") == 0.5
    assert parse_age("18 mo") == 1.5
    assert abs((parse_age("15 days") or 0) - 15 / 365.25) < 1e-9
    assert abs((parse_age("2 weeks") or 0) - 14 / 365.25) < 1e-9
    assert parse_age("1 year") == 1.0
    assert parse_age(None) is None
    assert parse_age("Positive") is None
    assert parse_age("") is None


def test_match_key_ignores_case_and_all_whitespace():
    # The curated aliases differ by spacing alone, and so do labs' test names.
    assert match_key("HDL / LDL Ratio") == match_key("hdl/ldl ratio") == "hdl/ldlratio"
    assert match_key("BILIRUBIN -DIRECT") == match_key("Bilirubin - Direct")
    assert match_key(None) == ""


# --- age brackets -----------------------------------------------------------
_ROWS = [(0, 150, 1.0, 9.0), (18, 60, 4.0, 6.0), (0, 1, 2.0, 3.0)]


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (40, (18, 60, 4.0, 6.0)),  # narrowest covering bracket wins
        (0.5, (0, 1, 2.0, 3.0)),  # a 6-month-old lands in the infant bracket
        (18, (18, 60, 4.0, 6.0)),  # inclusive lower bound
        (60, (18, 60, 4.0, 6.0)),  # inclusive upper bound
        (70, (0, 150, 1.0, 9.0)),  # only the catch-all covers this age
    ],
)
def test_pick_bracket_prefers_the_most_specific(age, expected):
    assert pick_bracket(_ROWS, age) == expected


def test_pick_bracket_needs_an_age_and_a_covering_row():
    # An adult range must never be applied to a patient whose age the report didn't give.
    assert pick_bracket(_ROWS, None) is None
    assert pick_bracket([], 40) is None
    assert pick_bracket([(18, 60, 4.0, 6.0)], 5) is None


def test_pick_bracket_is_deterministic_on_equal_spans():
    # Overlapping brackets of the same width: the lower one wins, every time.
    rows = [(20, 30, 5.0, 6.0), (10, 20, 3.0, 4.0)]
    assert pick_bracket(rows, 20) == (10, 20, 3.0, 4.0)


# --- units ------------------------------------------------------------------
def _lookup() -> Lookup:
    return Lookup(
        thp_id_by_name={"hemoglobin": 1, "hb": 1, "glucose": 2, "esr": 3},
        approved={1: True, 2: False, 3: True},
        canonical_name={1: "Hemoglobin", 2: "Glucose", 3: "ESR"},
        base_unit={1: "g/dl", 2: "mg/dl", 3: "mm/hr"},
        alt_units={(1, "g/l"): (0.1, 0.0), (2, "mmol/l"): (18.0, 0.0)},
        age_ranges={1: [(18, 60, 13.0, 17.0)], 3: [(0, 150, 0.0, 20.0)]},
    )


def test_in_report_unit_passes_the_base_unit_through():
    lk = _lookup()
    assert in_report_unit((13.0, 17.0), "g/dL", 1, lk) == (13.0, 17.0)
    assert in_report_unit((13.0, 17.0), "G/DL", 1, lk) == (13.0, 17.0)  # canonicalised


def test_in_report_unit_assumes_the_base_unit_when_none_is_printed():
    # Ratios, indices and counts often print no unit at all; refusing those would lose the
    # override for exactly the parameters whose unit a lab never bothers to print.
    lk = _lookup()
    assert in_report_unit((13.0, 17.0), None, 1, lk) == (13.0, 17.0)
    assert in_report_unit((13.0, 17.0), "", 1, lk) == (13.0, 17.0)


def test_in_report_unit_converts_a_curated_alternate():
    # Curated as printed -> base (base = printed * 0.1), so the bounds invert into g/L.
    lk = _lookup()
    assert in_report_unit((13.0, 17.0), "g/L", 1, lk) == (130.0, 170.0)


def test_in_report_unit_refuses_an_uncurated_unit():
    # Comparing 13-17 g/dL against a value printed in mmol/L would flag confidently wrong.
    lk = _lookup()
    assert in_report_unit((13.0, 17.0), "mmol/L", 1, lk) is None


def test_in_report_unit_keeps_bounds_ordered():
    lk = _lookup()
    low, high = in_report_unit((4.0, 6.0), "mmol/L", 2, lk) or (0.0, 0.0)
    assert low < high


# --- resolve ----------------------------------------------------------------
def test_resolve_exact_and_alias_hit():
    lk = _lookup()
    exact = resolve("Hemoglobin", "g/dL", 40, lk)
    assert exact.bounds == (13.0, 17.0)
    assert exact.source == "ideal_range"
    assert exact.matched_parameter == "Hemoglobin"
    assert exact.matched_group == "18-60"
    assert exact.reason is None
    assert resolve("HB", "g/dL", 40, lk).bounds == (13.0, 17.0)  # alias resolves to same


def test_resolve_unapproved_and_unmatched():
    lk = _lookup()
    assert resolve("Glucose", "mg/dL", 40, lk).reason == "unapproved"
    assert resolve("Glucose", "mg/dL", 40, lk).bounds is None
    assert resolve("Unknown Test", None, 40, lk).reason == "unmatched"
    assert resolve("Unknown Test", None, 40, lk).matched_parameter is None


def test_resolve_no_bracket_for_the_age_falls_back():
    lk = _lookup()
    # Hemoglobin is only curated for 18-60; a 5-year-old finds nothing.
    child = resolve("Hemoglobin", "g/dL", 5, lk)
    assert child.bounds is None
    assert child.reason == "no_ideal_range"
    # Same when the report never gave an age.
    assert resolve("Hemoglobin", "g/dL", None, lk).reason == "no_ideal_range"


def test_resolve_uses_an_age_agnostic_bracket():
    lk = _lookup()
    # ESR is curated 0-150, so it applies at any age.
    assert resolve("ESR", "mm/hr", 40, lk).bounds == (0.0, 20.0)
    assert resolve("ESR", "mm/hr", 0.5, lk).bounds == (0.0, 20.0)


def test_resolve_reports_a_unit_mismatch_with_the_bracket_it_reached():
    lk = _lookup()
    res = resolve("Hemoglobin", "mmol/L", 40, lk)
    assert res.reason == "unit_mismatch"
    assert res.bounds is None
    assert res.source == "report_range"
    # The bracket resolved; only the unit stopped it — R&D needs both to fix the curation.
    assert res.matched_parameter == "Hemoglobin"
    assert res.matched_group == "18-60"
