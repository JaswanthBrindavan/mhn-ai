"""Deterministic normalisation: abnormal flags and curated unit conversion.

This is the arithmetic CLAUDE.md says application code must own, so it is tested
directly and exhaustively for the parse formats and the conversion table.
"""

import pytest

from app.services.normalization import (
    abnormal_flag,
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
