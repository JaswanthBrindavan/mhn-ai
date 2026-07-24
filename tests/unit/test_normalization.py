"""Deterministic normalisation: abnormal flags and curated unit conversion.

This is the arithmetic CLAUDE.md says application code must own, so it is tested
directly and exhaustively for the parse formats and the conversion table.
"""

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
