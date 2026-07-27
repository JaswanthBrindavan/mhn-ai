"""Deterministic normalisation of extracted lab results — computed in Python, never by
the model (CLAUDE.md: don't ask the LLM for arithmetic application code can verify).

Two independent enrichments per result:

* ``abnormal_flag``  — parse the numeric value and the reference range (both in the
  source unit) and compare: ``low`` / ``normal`` / ``high``, or ``None`` when either
  side can't be parsed cleanly. Never guessed.
* unit conversion    — a small *curated* table (implementation.md §6), not a unit
  engine. A compatible unit yields ``normalized_value`` + ``normalized_unit`` with
  ``normalized: true``; anything outside the table stays as-extracted, ``normalized:
  false``.

# ponytail: curated conversion table + regex range parser, not a general unit/LOINC
# engine. Add pint / LOINC mapping if the analyte coverage ever needs to be broad.
"""

import re
from typing import Any

#: mg/dL -> mmol/L is analyte-dependent (molar mass). Scoped to the two analytes §6
#: names; every other mg/dL result is left un-normalised rather than mis-converted.
_MGDL_TO_MMOL = {
    "glucose": 0.0555,  # 1 / 18.0182
    "cholesterol": 0.02586,  # 1 / 38.67 (also HDL/LDL cholesterol)
}

#: Analyte-independent conversions, keyed by canonicalised source unit.
_UNIT_FACTORS: dict[str, tuple[str, float]] = {
    "g/dl": ("g/L", 10.0),
    "/ul": ("10^9/L", 0.001),  # e.g. 7000 /µL = 7 x10^9/L
}

_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
# low-high, allowing spaces and hyphen / en-dash / em-dash as the separator.
_RANGE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*[-–—]\s*(-?\d+(?:\.\d+)?)$")  # noqa: RUF001
_UPPER_RE = re.compile(r"^[<≤]=?\s*(-?\d+(?:\.\d+)?)$")  # < or <=
_LOWER_RE = re.compile(r"^[>≥]=?\s*(-?\d+(?:\.\d+)?)$")  # > or >=


def parse_number(value: str | None) -> float | None:
    """A clean scalar, or None. A comparator ('<0.01') or text ('Positive') → None."""
    if value is None:
        return None
    cleaned = value.strip().replace(",", "")
    return float(cleaned) if _NUMBER_RE.match(cleaned) else None


def parse_reference_range(text: str | None) -> tuple[float | None, float | None] | None:
    """(low, high) bounds, either possibly None for a one-sided range. None if unparseable."""
    if text is None:
        return None
    s = text.strip()
    if m := _RANGE_RE.match(s):
        return float(m.group(1)), float(m.group(2))
    if m := _UPPER_RE.match(s):
        return None, float(m.group(1))
    if m := _LOWER_RE.match(s):
        return float(m.group(1)), None
    return None


def abnormal_flag(
    value: float | None, bounds: tuple[float | None, float | None] | None
) -> str | None:
    """'low' | 'normal' | 'high', or None when it can't be determined."""
    if value is None or bounds is None:
        return None
    low, high = bounds
    if low is not None and value < low:
        return "low"
    if high is not None and value > high:
        return "high"
    return "normal"


def convert_unit(test_name: str, value: float | None, unit: str | None) -> tuple[float, str] | None:
    """Curated conversion to a canonical unit, or None if not covered."""
    if value is None or not unit:
        return None
    canon = _canon_unit(unit)

    if canon == "mg/dl":
        name = (test_name or "").lower()
        for analyte, factor in _MGDL_TO_MMOL.items():
            if analyte in name:
                return round(value * factor, 4), "mmol/L"
        return None

    if canon in _UNIT_FACTORS:
        target, factor = _UNIT_FACTORS[canon]
        return round(value * factor, 4), target
    return None


def enrich_result(
    result: dict[str, Any],
    *,
    override_bounds: tuple[float | None, float | None] | None = None,
    range_source: str = "report_range",
    matched_parameter: str | None = None,
    matched_group: str | None = None,
) -> dict[str, Any]:
    """Add the deterministic fields to one extracted result. Pure; returns a new dict.

    ``override_bounds`` is the R&D-approved ideal range for the patient's age group; when
    given it drives the abnormal flag instead of the report's printed reference range, and
    ``range_source`` is forced to ``"ideal_range"``. Without it, behaviour is unchanged
    (bounds parsed from ``reference_range``, ``range_source`` stays ``"report_range"``)."""
    value = parse_number(result.get("value"))
    if override_bounds is not None:
        bounds: tuple[float | None, float | None] | None = override_bounds
        range_source = "ideal_range"
    else:
        bounds = parse_reference_range(result.get("reference_range"))
    conv = convert_unit(result.get("test_name", ""), value, result.get("unit"))

    return {
        **result,
        "value_numeric": value,
        "abnormal_flag": abnormal_flag(value, bounds),
        "range_source": range_source,
        "matched_parameter": matched_parameter,
        "matched_group": matched_group,
        "normalized_value": conv[0] if conv else None,
        "normalized_unit": conv[1] if conv else None,
        "normalized": conv is not None,
    }


def _canon_unit(unit: str) -> str:
    """Canonicalise a unit string to ASCII so equivalent spellings match: lowercase,
    strip spaces, and fold the micro sign, superscript nine, and multiplication sign."""
    u = unit.strip().lower().replace(" ", "")
    u = u.replace("µ", "u").replace("μ", "u").replace("×", "x").replace("·", "")  # noqa: RUF001
    u = u.replace("⁹", "9").replace("^", "").replace("x109", "109").replace("*", "")
    return u


if __name__ == "__main__":  # pragma: no cover - self-check
    assert abnormal_flag(6.0, parse_reference_range("3.5-5.0")) == "high"
    assert abnormal_flag(2.0, parse_reference_range("3.5 - 5.0")) == "low"
    assert abnormal_flag(4.0, parse_reference_range("3.5–5.0")) == "normal"  # noqa: RUF001
    assert abnormal_flag(250, parse_reference_range("< 200")) == "high"
    assert abnormal_flag(50, parse_reference_range(">= 40")) == "normal"
    assert abnormal_flag(30, parse_reference_range("> 40")) == "low"
    assert parse_number("Positive") is None
    assert parse_number("<0.01") is None
    assert parse_number("5,200") == 5200.0
    assert abnormal_flag(parse_number("Positive"), parse_reference_range("3-5")) is None
    assert convert_unit("Fasting Glucose", 90, "mg/dL") == (round(90 * 0.0555, 4), "mmol/L")
    assert convert_unit("Hemoglobin", 14, "g/dL") == (140.0, "g/L")
    assert convert_unit("WBC", 7000, "/µL") == (7.0, "10^9/L")
    assert convert_unit("Sodium", 140, "mmol/L") is None  # outside the table
    # override_bounds wins over the report range: 100 is normal for 70-99? no -> high,
    # but the approved ideal (70, 90) also says high; prove the override drives it by
    # using an ideal range that DISAGREES with the report range.
    _report = {"value": "95", "reference_range": "70-99", "unit": None, "test_name": "Glucose"}
    assert enrich_result(_report)["abnormal_flag"] == "normal"  # report range: 95 in 70-99
    _over = enrich_result(_report, override_bounds=(70.0, 90.0), matched_group="adult all")
    assert _over["abnormal_flag"] == "high"  # ideal range 70-90: 95 is high
    assert _over["range_source"] == "ideal_range"
    assert _over["matched_group"] == "adult all"
