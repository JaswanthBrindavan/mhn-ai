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


#: "Male: 65-175, Female: 50-170" — one printed range carrying both sexes.
_GENDER_SEGMENT_RE = re.compile(r"\b(?P<gender>male|female)\s*:\s*(?P<range>[^,;]+)", re.I)
#: A leading qualifier ("Desirable : 2.5-3.0", "Adult : 17-43") or a trailing one
#: (">= 90 : Normal"). The label is context for a human, noise for the bounds.
_LEADING_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z ]{0,24}:\s*")
_TRAILING_LABEL_RE = re.compile(r"\s*:\s*[A-Za-z][A-Za-z ]{0,24}$")
#: Ratios print as "12:1 - 20:1"; the ":1" is notation, not a bound.
_RATIO_RE = re.compile(r"(\d+(?:\.\d+)?):1")
#: Words some labs use instead of a comparator.
_WORD_COMPARATORS = ((r"^below\s+", "< "), (r"^up\s*to\s+", "<= "), (r"^above\s+", "> "))
#: "70 to 100" is the same range as "70-100". Applied after the word comparators so
#: "Up to 40" has already become "<= 40" and is not mangled into a two-sided range.
_WORD_SEPARATOR_RE = re.compile(r"(\d)\s+to\s+(\d)", re.I)
#: The numeric core at the start of a cleaned range, ignoring any trailing unit
#: ("45-129U/L" -> "45-129", "90 - 120 mg/dl" -> "90 - 120").
_CORE_RE = re.compile(
    r"^(?:[<>≤≥]=?\s*-?\d+(?:\.\d+)?|-?\d+(?:\.\d+)?\s*[-–—]\s*-?\d+(?:\.\d+)?)"  # noqa: RUF001
)


def _select_gender_range(text: str, gender: str | None) -> str | None:
    """The segment for this patient's sex, or None when the choice can't be made.

    A gender-split range is unusable without knowing the sex — returning either half
    would flag against the wrong bounds, so an unknown sex stays unparsed.
    """
    segments: list[tuple[str, str]] = _GENDER_SEGMENT_RE.findall(text)
    if not segments:
        return text
    if not gender:
        return None
    wanted = "female" if gender.strip().lower().startswith("f") else "male"
    for found, range_text in segments:
        if found.lower() == wanted:
            return range_text.strip()
    return None


def _clean_range_text(text: str) -> str:
    """Strip the decoration labs print around bounds, leaving the numeric core."""
    s = text.strip()
    s = _TRAILING_LABEL_RE.sub("", s)
    if not _GENDER_SEGMENT_RE.match(s):  # a gender label is selected, never stripped
        s = _LEADING_LABEL_RE.sub("", s)
    s = _RATIO_RE.sub(r"\1", s)
    for pattern, replacement in _WORD_COMPARATORS:
        s = re.sub(pattern, replacement, s, flags=re.I)
    s = _WORD_SEPARATOR_RE.sub(r"\1-\2", s)
    if m := _CORE_RE.match(s.strip()):  # drops a trailing unit
        return m.group(0).strip()
    return s.strip()


def parse_reference_range(
    text: str | None, gender: str | None = None
) -> tuple[float | None, float | None] | None:
    """(low, high) bounds, either possibly None for a one-sided range. None if unparseable.

    ``gender`` selects the right half of a sex-split range ("Male: 65-175, Female:
    50-170"); without it such a range is left unparsed rather than guessed.
    """
    if text is None:
        return None
    selected = _select_gender_range(text.strip(), gender)
    if selected is None:
        return None
    s = _clean_range_text(selected)
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


_COMPARATOR_RE = re.compile(r"^(?P<op>[<>≤≥]=?)\s*(?P<number>-?\d+(?:[\d,]*\d)?(?:\.\d+)?)$")


def parse_comparator(value: str | None) -> tuple[str, float] | None:
    """A censored result like '< 148' or '>50.00' as (operator, bound), else None.

    Assays report beyond their measuring range as a comparator. The true value is unknown,
    so it never becomes a number — but the direction is often enough to decide the flag.
    """
    if value is None:
        return None
    m = _COMPARATOR_RE.match(value.strip().replace(",", ""))
    if m is None:
        return None
    op = m.group("op").replace("≤", "<=").replace("≥", ">=")
    return op, float(m.group("number"))


def comparator_flag(
    value: str | None, bounds: tuple[float | None, float | None] | None
) -> str | None:
    """Flag a censored value, but only when EVERY value it could stand for agrees.

    '< 148' against 187-833 is low: whatever the true value is, it is under 148, which is
    already under 187. '< 200' against 187-833 is indeterminate — the real value could be
    190 (normal) or 100 (low) — so it stays None rather than guessing.
    """
    parsed = parse_comparator(value)
    if parsed is None or bounds is None:
        return None
    op, x = parsed
    low, high = bounds

    if op in ("<", "<="):
        # Below the bound. Certain only if the bound sits at or under the low limit.
        if low is not None and (x <= low if op == "<" else x < low):
            return "low"
        # With no low limit the range is one-sided, so being under the high limit is normal.
        if low is None and high is not None and x <= high:
            return "normal"
        return None

    if high is not None and (x >= high if op == ">" else x > high):
        return "high"
    if high is None and low is not None and x >= low:
        return "normal"
    return None


#: Words meaning "none detected". Against a numeric range they read as zero — a urine
#: glucose of "Nil" against "0 - 2" is in range, and saying so is arithmetic, not judgement.
_ABSENT_TERMS = frozenset({"nil", "none", "negative", "absent", "not detected", "nd", "no"})
#: Words meaning "found". Prefixes, because labs qualify them: "Present 3+(500-1000 mg/dl)".
#: Checked with startswith so "not detected" is never read as "detected".
_PRESENT_PREFIXES = ("present", "positive", "detected", "reactive", "seen")
#: The report stating its own verdict. "Urobilinogen: Normal" needs no arithmetic.
_IN_RANGE_TERMS = frozenset({"normal", "within normal limits", "wnl", "within normal range"})
#: A result printed as its own small range, e.g. a microscopy count of "1-2".
_VALUE_RANGE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*[-–—]\s*(-?\d+(?:\.\d+)?)$")  # noqa: RUF001


def _norm_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def qualitative_flag(
    value: str | None,
    reference_range: str | None,
    bounds: tuple[float | None, float | None] | None,
) -> str | None:
    """Flag a non-numeric result, only where the comparison is unambiguous.

    The safe cases: the result matches what the range says to expect ("Negative" vs
    "Negative"); the report calls it normal itself; something is present where the range
    expects absence; an "absent" word against a numeric range, which is zero; and a result
    printed as its own range ("1-2" against "0-5"), where both ends decide it together.
    Anything else — "1-2" against "Few" — stays None rather than guessing.

    ``bounds`` is passed in rather than re-parsed so an approved ideal range still wins.
    """
    if value is None or reference_range is None:
        return None
    val, ref = _norm_text(value), _norm_text(reference_range)
    if not val or not ref:
        return None

    if val == ref or (val in _ABSENT_TERMS and ref in _ABSENT_TERMS):
        return "normal"
    if val in _IN_RANGE_TERMS:
        return "normal"
    if ref in _ABSENT_TERMS and val.startswith(_PRESENT_PREFIXES):
        # Detected where nothing should be: out of range, and the report's own "+" grading
        # is not a number we can compare, so "high" is as specific as we may be.
        return "high"

    if bounds is None:
        return None

    if val in _ABSENT_TERMS:
        # Only where "none" is a sensible reading of the range — one that starts at zero
        # or has no floor. A range like 187-833 belongs to a quantitative analyte, so a
        # qualitative result against it is contradictory data, not a low value.
        low, _ = bounds
        return abnormal_flag(0.0, bounds) if low in (None, 0.0) else None

    if m := _VALUE_RANGE_RE.match(val):
        low_flag = abnormal_flag(float(m.group(1)), bounds)
        high_flag = abnormal_flag(float(m.group(2)), bounds)
        # Only decide when both ends land in the same place; a straddling count is unknown.
        return low_flag if low_flag == high_flag else None

    return None


def convert_unit(test_name: str, value: float | None, unit: str | None) -> tuple[float, str] | None:
    """Curated conversion to a canonical unit, or None if not covered."""
    if value is None or not unit:
        return None
    canon = canon_unit(unit)

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
    gender: str | None = None,
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
        bounds = parse_reference_range(result.get("reference_range"), gender)
    conv = convert_unit(result.get("test_name", ""), value, result.get("unit"))

    # Values that are not plain numbers still carry decidable information. A censored
    # value ('< 148') has a direction; a qualitative one ('Negative', 'Nil', '1-2') can
    # match what the range expects. Keeping both in Python matters: otherwise the
    # comparison silently falls to the model, which the extraction prompt forbids.
    flag = abnormal_flag(value, bounds)
    if flag is None and value is None:
        flag = comparator_flag(result.get("value"), bounds) or qualitative_flag(
            result.get("value"), result.get("reference_range"), bounds
        )

    return {
        **result,
        "value_numeric": value,
        "abnormal_flag": flag,
        "range_source": range_source,
        "matched_parameter": matched_parameter,
        "matched_group": matched_group,
        "normalized_value": conv[0] if conv else None,
        "normalized_unit": conv[1] if conv else None,
        "normalized": conv is not None,
    }


def canon_unit(unit: str) -> str:
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
    # Censored values: flag only when every value the comparator could stand for agrees.
    assert parse_comparator("< 148") == ("<", 148.0)
    assert parse_comparator(">50.00") == (">", 50.0)
    assert parse_comparator("≤ 0.01") == ("<=", 0.01)
    assert parse_comparator("Negative") is None and parse_comparator("148") is None
    # Real cases from live reports.
    assert comparator_flag("< 148", parse_reference_range("187 - 833")) == "low"
    assert comparator_flag("> 50.00", parse_reference_range("5.46 - 16.20")) == "high"
    assert comparator_flag("<20.0", parse_reference_range("0 - 29.9")) is None  # could be 25
    # Indeterminate: the true value may sit either side of the limit.
    assert comparator_flag("< 200", parse_reference_range("187 - 833")) is None
    assert comparator_flag("> 190", parse_reference_range("187 - 833")) is None
    # One-sided ranges: under an upper limit is unambiguously normal.
    assert comparator_flag("< 0.01", parse_reference_range("< 5")) == "normal"
    assert comparator_flag("> 90", parse_reference_range(">= 60")) == "normal"
    # Strictness matters: '<= 187' allows exactly 187, which is in range.
    assert comparator_flag("<= 187", parse_reference_range("187 - 833")) is None
    assert comparator_flag("< 187", parse_reference_range("187 - 833")) == "low"
    # End to end through enrich_result, and the value itself is never invented.
    _b12 = {"value": "< 148", "reference_range": "187 - 833", "unit": None, "test_name": "B12"}
    assert enrich_result(_b12)["abnormal_flag"] == "low"
    assert enrich_result(_b12)["value_numeric"] is None
    assert enrich_result(_b12)["value"] == "< 148"

    # Decorated ranges: a trailing unit, a label, a ratio notation, a word comparator.
    assert parse_reference_range("90 - 120 mg/dl") == (90.0, 120.0)
    assert parse_reference_range("45-129U/L") == (45.0, 129.0)
    assert parse_reference_range("Desirable : 2.5-3.0") == (2.5, 3.0)
    assert parse_reference_range("Adult : 17-43") == (17.0, 43.0)
    assert parse_reference_range(">= 90 : Normal") == (90.0, None)
    assert parse_reference_range("12:1 - 20:1") == (12.0, 20.0)
    assert parse_reference_range("9:1-23:1") == (9.0, 23.0)
    assert parse_reference_range("Below 5.7%") == (None, 5.7)
    assert parse_reference_range("70 to 100") == (70.0, 100.0)
    assert parse_reference_range("Up to 40") == (None, 40.0)  # not mangled into a range
    # Sex-split ranges need the report's gender; without it, no guess.
    _split = "Male: 65-175, Female: 50-170"
    assert parse_reference_range(_split, "M") == (65.0, 175.0)
    assert parse_reference_range(_split, "Female") == (50.0, 170.0)
    assert parse_reference_range(_split) is None
    assert parse_reference_range("Male: 225 - 535 µg/dl, Female: 215 - 535 µg/dl", "M") == (
        225.0,
        535.0,
    )
    # Qualitative results.
    assert qualitative_flag("Negative", "Negative", None) == "normal"
    assert qualitative_flag("Pale Yellow", "pale yellow", None) == "normal"
    assert qualitative_flag("Nil", "<0.01", (None, 0.01)) == "normal"  # absent reads as 0
    assert qualitative_flag("Nil", "0 - 2", (0.0, 2.0)) == "normal"
    assert qualitative_flag("1-2", "0 - 5", (0.0, 5.0)) == "normal"  # both ends in range
    assert qualitative_flag("1-2", "Few", None) is None  # undecidable
    # Contradictory pairing: a quantitative range with a qualitative result decides nothing.
    assert qualitative_flag("Negative", "187 - 833", (187.0, 833.0)) is None
    # Present where the range expects absence, however the lab grades it.
    assert qualitative_flag("Present 3+(500-1000 mg/dl)", "Absent", None) == "high"
    assert qualitative_flag("PRESENT", "Absent", None) == "high"
    assert qualitative_flag("Positive", "Negative", None) == "high"
    assert qualitative_flag("Not detected", "Absent", None) == "normal"  # never read as present
    # The report stating its own verdict, against a numeric range it does not fill in.
    assert qualitative_flag("Normal", "<=0.2", (None, 0.2)) == "normal"
    assert qualitative_flag("WNL", "0 - 5", (0.0, 5.0)) == "normal"
    assert qualitative_flag("4-8", "0 - 5", (0.0, 5.0)) is None  # straddles the limit
    assert qualitative_flag("8-12", "0 - 5", (0.0, 5.0)) == "high"  # both ends above

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
