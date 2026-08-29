import pytest

from app.services.ideal_ranges import (
    Bracket,
    Lookup,
    canon_sex,
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


# --- sex --------------------------------------------------------------------
@pytest.mark.parametrize("raw", ["Male", "male", "M", "m", " MALE ", "boy"])
def test_canon_sex_reads_the_spellings_a_report_prints(raw):
    assert canon_sex(raw) == "male"


@pytest.mark.parametrize("raw", ["Female", "F", "woman", "girl"])
def test_canon_sex_reads_female(raw):
    assert canon_sex(raw) == "female"


@pytest.mark.parametrize("raw", [None, "", "Other", "Transgender", "U", "N/A", "X"])
def test_canon_sex_refuses_anything_it_does_not_recognise(raw):
    # Stricter than the printed-range selector on purpose: an unrecognised value must not
    # fall through to "male", which would flag a patient against the wrong curated range.
    assert canon_sex(raw) is None


# --- age brackets -----------------------------------------------------------
_ROWS = [
    Bracket(0, 150, 1.0, 9.0, "any"),
    Bracket(18, 60, 4.0, 6.0, "any"),
    Bracket(0, 1, 2.0, 3.0, "any"),
]


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (40, (18, 60, 4.0, 6.0, "any")),  # narrowest covering bracket wins
        (0.5, (0, 1, 2.0, 3.0, "any")),  # a 6-month-old lands in the infant bracket
        (18, (18, 60, 4.0, 6.0, "any")),  # inclusive lower bound
        (60, (18, 60, 4.0, 6.0, "any")),  # inclusive upper bound
        (70, (0, 150, 1.0, 9.0, "any")),  # only the catch-all covers this age
    ],
)
def test_pick_bracket_prefers_the_most_specific(age, expected):
    assert pick_bracket(_ROWS, age) == expected


def test_pick_bracket_needs_an_age_and_a_covering_row():
    # An adult range must never be applied to a patient whose age the report didn't give.
    assert pick_bracket(_ROWS, None) is None
    assert pick_bracket([], 40) is None
    assert pick_bracket([Bracket(18, 60, 4.0, 6.0, "any")], 5) is None


def test_pick_bracket_is_deterministic_on_equal_spans():
    # Overlapping brackets of the same width: the lower one wins, every time.
    rows = [Bracket(20, 30, 5.0, 6.0, "any"), Bracket(10, 20, 3.0, 4.0, "any")]
    assert pick_bracket(rows, 20) == (10, 20, 3.0, 4.0, "any")


# --- sex-specific brackets --------------------------------------------------
# The shape 78 of the 277 curated ranges actually have: a male row and a female row, no
# 'any' row at all (V18's unique key is (thp_id, sex, age_min, age_max)).
_SEXED = [Bracket(0, 120, 13.0, 17.0, "male"), Bracket(0, 120, 12.0, 15.0, "female")]


def test_pick_bracket_takes_the_patients_own_sex():
    assert pick_bracket(_SEXED, 40, "Male") == (0, 120, 13.0, 17.0, "male")
    assert pick_bracket(_SEXED, 40, "F") == (0, 120, 12.0, 15.0, "female")


def test_pick_bracket_takes_neither_half_when_the_sex_is_unknown():
    # The whole bug this guards: without reading `sex`, a woman was flagged against the
    # male range 13-17 because it merely sorted first. Nothing is better than the wrong one.
    assert pick_bracket(_SEXED, 40, None) is None
    assert pick_bracket(_SEXED, 40, "Other") is None


def test_pick_bracket_prefers_sex_over_a_narrower_age_span():
    # A parameter is curated per sex precisely when sex is what moves the range, so the
    # sex-specific row wins even against an 'any' row that covers a tighter age band.
    rows = [*_SEXED, Bracket(18, 60, 1.0, 2.0, "any")]
    assert pick_bracket(rows, 40, "male") == (0, 120, 13.0, 17.0, "male")


def test_pick_bracket_falls_back_to_any_when_the_sex_has_no_row():
    rows = [Bracket(0, 120, 5.0, 9.0, "any"), Bracket(0, 120, 13.0, 17.0, "male")]
    assert pick_bracket(rows, 40, "female") == (0, 120, 5.0, 9.0, "any")
    assert pick_bracket(rows, 40, None) == (0, 120, 5.0, 9.0, "any")


# --- units ------------------------------------------------------------------
def _lookup() -> Lookup:
    return Lookup(
        thp_id_by_name={"hemoglobin": 1, "hb": 1, "glucose": 2, "esr": 3},
        approved={1: True, 2: False, 3: True},
        canonical_name={1: "Hemoglobin", 2: "Glucose", 3: "ESR"},
        base_unit={1: "g/dl", 2: "mg/dl", 3: "mm/hr"},
        alt_units={(1, "g/l"): (0.1, 0.0), (2, "mmol/l"): (18.0, 0.0)},
        age_ranges={
            1: [Bracket(18, 60, 13.0, 17.0, "any")],
            3: [Bracket(0, 150, 0.0, 20.0, "any")],
        },
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


def _sexed_lookup() -> Lookup:
    lk = _lookup()
    return Lookup(
        thp_id_by_name={**lk.thp_id_by_name, "ferritin": 4},
        approved={**lk.approved, 4: True},
        canonical_name={**lk.canonical_name, 4: "Ferritin"},
        base_unit={**lk.base_unit, 4: "ng/ml"},
        alt_units=lk.alt_units,
        age_ranges={
            **lk.age_ranges,
            4: [Bracket(0, 120, 30.0, 400.0, "male"), Bracket(0, 120, 15.0, 150.0, "female")],
        },
    )


def test_resolve_uses_the_bracket_for_the_reports_own_sex():
    lk = _sexed_lookup()
    assert resolve("Ferritin", "ng/mL", 40, lk, sex="Female").bounds == (15.0, 150.0)
    assert resolve("Ferritin", "ng/mL", 40, lk, sex="M").bounds == (30.0, 400.0)


def test_resolve_falls_back_when_a_sex_split_parameter_has_no_sex():
    # Same rule as a printed 'Male: … Female: …' range: unusable without knowing which.
    res = resolve("Ferritin", "ng/mL", 40, lk := _sexed_lookup())
    assert res.bounds is None
    assert res.reason == "no_ideal_range"
    assert res.matched_parameter == "Ferritin"  # matched, just not resolvable
    assert resolve("Ferritin", "ng/mL", 40, lk, sex="Other").reason == "no_ideal_range"


def test_resolve_names_the_sex_in_the_group_only_when_it_narrowed_the_choice():
    # R&D reads this on the worklist: "18-60" and "0-120 female" are different curations.
    assert (
        resolve("Ferritin", "ng/mL", 40, _sexed_lookup(), sex="F").matched_group == "0-120 female"
    )
    assert resolve("Hemoglobin", "g/dL", 40, _lookup(), sex="F").matched_group == "18-60"


def test_resolve_reports_a_unit_mismatch_with_the_bracket_it_reached():
    lk = _lookup()
    res = resolve("Hemoglobin", "mmol/L", 40, lk)
    assert res.reason == "unit_mismatch"
    assert res.bounds is None
    assert res.source == "report_range"
    # The bracket resolved; only the unit stopped it — R&D needs both to fix the curation.
    assert res.matched_parameter == "Hemoglobin"
    assert res.matched_group == "18-60"
