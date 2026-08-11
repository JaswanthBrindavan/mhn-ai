"""Deterministic normalisation: abnormal flags and curated unit conversion.

This is the arithmetic CLAUDE.md says application code must own, so it is tested
directly and exhaustively for the parse formats and the conversion table.
"""

import pytest

from app.services.normalization import (
    abnormal_flag,
    category_split_flag,
    convert_unit,
    enrich_result,
    parse_number,
    parse_reference_range,
)


def _flag(value, ref):
    return abnormal_flag(parse_number(value), parse_reference_range(ref))


# --- abnormal flags ---------------------------------------------------------


def test_two_sided_range_flags():
    assert _flag("6.0", "3.5-5.0") == "high"
    assert _flag("2.0", "3.5-5.0") == "low"
    assert _flag("4.0", "3.5-5.0") == "normal"


def test_boundary_is_inclusive_normal():
    # On the boundary is not out of range.
    assert _flag("3.5", "3.5-5.0") == "normal"
    assert _flag("5.0", "3.5-5.0") == "normal"


def test_range_separators_and_spaces():
    assert _flag("4.0", "3.5 - 5.0") == "normal"
    assert _flag("4.0", "3.5–5.0") == "normal"  # en dash  # noqa: RUF001


def test_one_sided_ranges():
    assert _flag("250", "< 200") == "high"
    assert _flag("150", "<200") == "normal"
    assert _flag("30", "> 40") == "low"
    assert _flag("50", ">= 40") == "normal"


def test_unparseable_inputs_yield_no_flag():
    assert _flag("Positive", "3-5") is None  # non-numeric value
    assert _flag("<0.01", "0-5") is None  # comparator value, not a clean number
    assert _flag("4.0", "see notes") is None  # unparseable range
    assert _flag("4.0", None) is None


def test_thousands_separator_parses():
    assert parse_number("5,200") == 5200.0


# --- unit conversion --------------------------------------------------------


def test_analyte_dependent_mgdl_conversion():
    assert convert_unit("Fasting Glucose", 90, "mg/dL") == (round(90 * 0.0555, 4), "mmol/L")
    assert convert_unit("HDL Cholesterol", 60, "mg/dL") == (round(60 * 0.02586, 4), "mmol/L")


def test_mgdl_without_known_analyte_is_not_converted():
    assert convert_unit("Creatinine", 1.2, "mg/dL") is None


def test_analyte_independent_conversions():
    assert convert_unit("Hemoglobin", 14, "g/dL") == (140.0, "g/L")
    assert convert_unit("WBC", 7000, "/µL") == (7.0, "10^9/L")


def test_unit_spelling_is_canonicalised():
    # µ vs u, and casing, still match.
    assert convert_unit("WBC", 7000, "/uL") == (7.0, "10^9/L")


def test_unknown_unit_is_left_alone():
    assert convert_unit("Sodium", 140, "mmol/L") is None
    assert convert_unit("X", None, "mg/dL") is None


# --- enrichment glue --------------------------------------------------------


def test_enrich_result_adds_all_deterministic_fields():
    enriched = enrich_result(
        {
            "test_name": "Fasting Glucose",
            "value": "126",
            "unit": "mg/dL",
            "reference_range": "70-99",
        }
    )
    assert enriched["value_numeric"] == 126.0
    assert enriched["abnormal_flag"] == "high"
    assert enriched["normalized"] is True
    assert enriched["normalized_unit"] == "mmol/L"
    # Original fields are preserved.
    assert enriched["test_name"] == "Fasting Glucose"


def test_enrich_result_leaves_non_numeric_unflagged():
    enriched = enrich_result(
        {"test_name": "HIV", "value": "Non-reactive", "unit": None, "reference_range": None}
    )
    assert enriched["value_numeric"] is None
    assert enriched["abnormal_flag"] is None
    assert enriched["normalized"] is False


@pytest.mark.parametrize(
    ("value", "reference_range", "expected"),
    [
        # Certain: whatever the true value is, it lies outside the range.
        ("< 148", "187 - 833", "low"),  # Vitamin B12 below the assay's floor
        ("> 50.00", "5.46 - 16.20", "high"),  # Homocysteine above its ceiling
        ("< 187", "187 - 833", "low"),  # strictly under the low bound
        # Certain the other way: a one-sided range makes "under the limit" normal.
        ("< 0.01", "< 5", "normal"),
        ("> 90", ">= 60", "normal"),
        # Indeterminate: the true value could sit either side, so no guess is made.
        ("< 200", "187 - 833", None),  # could be 190 (normal) or 100 (low)
        ("<20.0", "0 - 29.9", None),  # could be 25, which is in range
        ("<= 187", "187 - 833", None),  # 187 itself is in range
        ("> 190", "187 - 833", None),
        # Not a comparator at all.
        ("Negative", "187 - 833", None),
        ("< 148", None, None),  # nothing to compare against
    ],
)
def test_censored_values_are_flagged_only_when_unambiguous(value, reference_range, expected):
    enriched = enrich_result(
        {"test_name": "Vitamin B12", "value": value, "reference_range": reference_range}
    )
    assert enriched["abnormal_flag"] == expected


def test_censored_value_is_flagged_without_inventing_a_number():
    """The flag comes from the comparator; the value stays exactly as printed."""
    enriched = enrich_result(
        {"test_name": "Vitamin B12", "value": "< 148", "reference_range": "187 - 833"}
    )
    assert enriched["abnormal_flag"] == "low"
    assert enriched["value_numeric"] is None
    assert enriched["value"] == "< 148"


def test_override_bounds_apply_to_censored_values_too():
    """An approved ideal range drives a censored value's flag, same as a numeric one."""
    result = {"test_name": "Vitamin B12", "value": "< 148", "reference_range": "100 - 800"}
    assert enrich_result(result)["abnormal_flag"] is None  # 148 sits inside 100-800
    enriched = enrich_result(result, override_bounds=(187.0, 833.0))
    assert enriched["abnormal_flag"] == "low"
    assert enriched["range_source"] == "ideal_range"


@pytest.mark.parametrize(
    ("reference_range", "expected"),
    [
        ("90 - 120 mg/dl", (90.0, 120.0)),  # trailing unit
        ("45-129U/L", (45.0, 129.0)),  # unit glued to the bound
        ("Desirable : 2.5-3.0", (2.5, 3.0)),  # leading label
        ("Adult : 17-43", (17.0, 43.0)),
        (">= 90 : Normal", (90.0, None)),  # trailing label
        ("12:1 - 20:1", (12.0, 20.0)),  # ratio notation
        ("9:1-23:1", (9.0, 23.0)),
        ("Below 5.7%", (None, 5.7)),  # word comparator + unit
        ("Up to 40", (None, 40.0)),  # "to" here is part of the comparator, not a separator
        ("Above 60", (60.0, None)),
        ("70 to 100", (70.0, 100.0)),  # "to" as the separator
        ("70 TO 100 mg/dl", (70.0, 100.0)),
        ("3.5-5.0", (3.5, 5.0)),  # plain forms still work
        ("Few", None),  # genuinely not a numeric range
    ],
)
def test_decorated_reference_ranges_are_parsed(reference_range, expected):
    assert parse_reference_range(reference_range) == expected


def test_gender_split_range_needs_the_reports_gender():
    """Picking the wrong half would flag against the wrong bounds, so unknown means unparsed."""
    split = "Male: 65-175, Female: 50-170"
    assert parse_reference_range(split, "M") == (65.0, 175.0)
    assert parse_reference_range(split, "Female") == (50.0, 170.0)
    assert parse_reference_range(split) is None
    assert parse_reference_range(split, "") is None
    # Iron: 74 is in range for a man, high-normal for a woman — the choice matters.
    male = enrich_result({"test_name": "Iron", "value": "60", "reference_range": split}, gender="M")
    female = enrich_result(
        {"test_name": "Iron", "value": "60", "reference_range": split}, gender="F"
    )
    assert male["abnormal_flag"] == "low"  # under 65
    assert female["abnormal_flag"] == "normal"  # over 50


@pytest.mark.parametrize(
    ("value", "reference_range", "expected"),
    [
        ("Negative", "Negative", "normal"),  # says what it should
        ("Nil", "Nil", "normal"),
        ("Pale Yellow", "pale yellow", "normal"),  # case and spacing folded
        ("Nil", "<0.01", "normal"),  # absent reads as zero
        ("Nil", "0 - 2", "normal"),
        ("1-2", "0 - 5", "normal"),  # count range inside the limits
        ("8-12", "0 - 5", "high"),  # entirely above
        ("4-8", "0 - 5", None),  # straddles the limit — undecidable
        ("1-2", "Few", None),  # non-numeric expectation
        # Detected where the range expects nothing, however the lab grades it.
        ("Present 3+(500-1000 mg/dl)", "Absent", "high"),
        ("PRESENT", "Absent", "high"),
        ("Positive", "Negative", "high"),
        ("Not detected", "Absent", "normal"),  # startswith("not") — never read as present
        ("Present", "Present", "normal"),  # present is what this range expects
        # The report gives its own verdict against a range it does not fill in.
        ("Normal", "<=0.2", "normal"),
        ("WNL", "0 - 5", "normal"),
    ],
)
def test_qualitative_results_are_flagged_only_when_unambiguous(value, reference_range, expected):
    enriched = enrich_result(
        {"test_name": "Urine Glucose", "value": value, "reference_range": reference_range}
    )
    assert enriched["abnormal_flag"] == expected


def test_enrich_result_defaults_to_report_range_source():
    enriched = enrich_result({"test_name": "Glucose", "value": "95", "reference_range": "70-99"})
    assert enriched["abnormal_flag"] == "normal"  # 95 in 70-99
    assert enriched["range_source"] == "report_range"
    assert enriched["matched_parameter"] is None
    assert enriched["matched_group"] is None


def test_override_bounds_win_over_report_range():
    result = {"test_name": "Glucose", "value": "95", "reference_range": "70-99"}
    # Report range says 95 is normal; the approved ideal range (70-90) says high.
    enriched = enrich_result(
        result,
        override_bounds=(70.0, 90.0),
        matched_parameter="Fasting Glucose",
        matched_group="adult male",
    )
    assert enriched["abnormal_flag"] == "high"
    assert enriched["range_source"] == "ideal_range"
    assert enriched["matched_parameter"] == "Fasting Glucose"
    assert enriched["matched_group"] == "adult male"


def test_sex_split_range_without_a_comma_separator() -> None:
    """Labs separate the two halves with nothing but a space.

    The original pattern ended a segment only at ',' or ';', so the male half greedily
    swallowed the female half and the female segment never existed — leaving a genuinely
    abnormal result unflagged. Measured on a real report: SERUM COPPER 162.22 against
    "Male : 63.5 - 150 Female : 80 - 155" for a female patient came back with no flag.
    """
    copper = "Male : 63.5 - 150 Female : 80 - 155"
    assert parse_reference_range(copper, "F") == (80.0, 155.0)
    assert parse_reference_range(copper, "M") == (63.5, 150.0)

    iron = "Male : 65 - 175 Female : 50 - 170"
    assert parse_reference_range(iron, "Female") == (50.0, 170.0)

    # Trailing units on each half, still no comma.
    tibc = "Male: 225 - 535 µg/dl Female: 215 - 535 µg/dl"
    assert parse_reference_range(tibc, "F") == (215.0, 535.0)

    # The comma-separated form must keep working.
    assert parse_reference_range("Male: 65-175, Female: 50-170", "F") == (50.0, 170.0)
    # Unknown sex still refuses to pick a half.
    assert parse_reference_range(copper, None) is None


def test_sex_split_flags_the_real_copper_and_iron_results() -> None:
    """End to end through enrich_result, the way extraction calls it."""
    copper = enrich_result(
        {
            "test_name": "SERUM COPPER",
            "value": "162.22",
            "unit": "µg/dL",
            "reference_range": "Male : 63.5 - 150 Female : 80 - 155",
        },
        gender="F",
    )
    assert copper["abnormal_flag"] == "high"

    iron = enrich_result(
        {
            "test_name": "IRON",
            "value": "31",
            "unit": "µg/dL",
            "reference_range": "Male : 65 - 175 Female : 50 - 170",
        },
        gender="F",
    )
    assert iron["abnormal_flag"] == "low"


# --- interpretation scales and category splits ------------------------------


@pytest.mark.parametrize(
    ("reference_range", "expected"),
    [
        # eGFR, the shape that prompted this: normal band printed first.
        (">= 90 : Normal 60 - 89 : Mild Decrease 45 - 59 : Moderate Decrease", (90.0, None)),
        # The dangerous shape: normal is NOT first, so taking the first bound was wrong.
        ("Low: <70 Normal: 70-99 High: >=100", (70.0, 99.0)),
        ("Deficiency: <20 Insufficiency: 20-29 Sufficiency: 30-100", (30.0, 100.0)),
        ("Normal: <5.7 Prediabetes: 5.7-6.4 Diabetes: >=6.5", (None, 5.7)),
        # No grade means "in range", so there is nothing to select.
        ("Grade I: <10 Grade II: 10-20 Grade III: >20", None),
        # A single labelled range is not a scale and keeps its existing reading.
        ("Desirable : 2.5-3.0", (2.5, 3.0)),
        (">= 90 : Normal", (90.0, None)),
    ],
)
def test_multi_band_scales_are_read_by_their_normal_band(reference_range, expected):
    assert parse_reference_range(reference_range) == expected


def test_a_scale_with_normal_in_the_middle_no_longer_flags_healthy_values():
    """The bug this fixes produced a WRONG flag, not a missing one.

    Reading the first bound of "Low: <70 Normal: 70-99 High: >=100" gave bounds of
    (None, 70), so a value of 85 — squarely normal — came back "high". The same shape
    reported a healthy vitamin D of 45 as high against the deficiency band.
    """
    scale = "Low: <70 Normal: 70-99 High: >=100"
    assert _flag("85", scale) == "normal"
    assert _flag("50", scale) == "low"
    assert _flag("120", scale) == "high"

    vitamin_d = "Deficiency: <20 Insufficiency: 20-29 Sufficiency: 30-100"
    assert _flag("45", vitamin_d) == "normal"
    assert _flag("12", vitamin_d) == "low"


#: CEA prints one threshold per smoking status, and the report never says which applies.
_CEA = "Non Smokers (Past / Never Smoked) - <5 Smokers (current) - <10"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.93, "normal"),  # under both thresholds, so normal either way
        (4.9, "normal"),
        (12.0, "high"),  # over both
        (7.0, None),  # between them: depends on a category we were not told
        (5.0, None),
    ],
)
def test_a_category_split_decides_only_when_every_threshold_agrees(value, expected):
    assert category_split_flag(value, _CEA) == expected


def test_a_single_threshold_is_not_a_category_split():
    """One comparator is an ordinary one-sided range, read properly elsewhere."""
    assert category_split_flag(3.0, "< 5") is None


def test_thresholds_pointing_opposite_ways_are_refused():
    """Those describe bands, not one boundary drawn twice, so nothing can be concluded
    from 'all agree'."""
    assert category_split_flag(85.0, "Low: <70 High: >=100") is None


def test_the_category_split_reaches_a_real_result():
    """Wired into enrich_result, which is what actually stores the flag."""
    enriched = enrich_result(
        {"test_name": "CEA", "value": "0.93", "unit": "ng/mL", "reference_range": _CEA}
    )
    assert enriched["abnormal_flag"] == "normal"


def test_not_done_is_never_read_as_not_detected() -> None:
    """ "ND" means both, and they are opposites here.

    Read as absent, it becomes 0.0 against the range and comes back `normal` — a clean
    result published for an assay nobody ran, on a report a person will act on. Every
    other decision in this module refuses when the readings disagree; a two-letter
    abbreviation is not the place to start making exceptions.

    The cost is a genuine "Not Detected" going unflagged, which is recoverable.
    """
    row = {"test_name": "Urine Glucose", "value": "ND", "unit": None, "reference_range": "0 - 2"}

    assert enrich_result(row)["abnormal_flag"] is None

    # The unambiguous spellings still work, so nothing real was lost.
    for spelling in ("Nil", "Absent", "Not Detected", "Negative"):
        checked = enrich_result({**row, "value": spelling})
        assert checked["abnormal_flag"] == "normal", spelling
