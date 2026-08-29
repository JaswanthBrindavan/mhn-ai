"""Resolve the doctor-approved ideal range for a lab result, by the patient's age.

For each extracted test we try to match it to an R&D-approved THP (traditional health
parameter) and pick the ideal range for the patient's age bracket; that range then
overrides the report's printed reference range for the abnormal-flag calculation. A miss —
the test is not a known parameter, the parameter is not approved, no bracket covers the
age, or the report printed a unit we cannot convert — leaves the report's own range in
charge and is recorded as an R&D worklist fallback.

All logic here is deterministic Python. The Spring-owned THP tables are read once per
document (``load_lookup``); everything else is pure and unit-testable without a DB.

# ponytail: three dict lookups over the curated tables, no demographics engine. The one
# unconfirmed thing left is the direction of thp_alternate_units.multiplier — see
# ``in_report_unit``.
"""

import re
from dataclasses import dataclass
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.spring import (
    thp_age_range,
    thp_alias,
    thp_alternate_units,
    traditional_health_parameters,
)
from app.services.normalization import canon_unit

# --- what counts as "in range" ----------------------------------------------
# Three zones, since Spring's V28 dropped the danger tiers:
#
#   min ──warning low── low_warn ══ ideal band (marker at `ideal`) ══ high_warn
#       ──warning high── max
#
# The abnormal flag is computed against (low_warn, high_warn) — the pair the staff
# dashboard labels "Ideal Range". That reading is unchanged by V28, and is now the whole
# of the model rather than the middle of it.
#
# ``min``/``max`` are the GRAPH bounds and are not read. Note this reverses what R&D said
# in 2026-07-31 (that they were the outer ends of the danger zones, i.e. clinical
# boundaries); V28's own comment settles it the other way, and the dashboard never
# collected danger values. ``ideal`` is the target *inside* the band, not an edge of it —
# the CHECK pins it there — so nothing here reads that either.
_IDEAL_FLOOR = thp_age_range.c.low_warn
_IDEAL_CEILING = thp_age_range.c.high_warn

#: The one value of Spring's ``reference_status_enum`` that lets curated data decide a
#: patient's flag. Everything else — draft, pending, rejected, archived, merged — is work
#: in progress on the staff dashboard and must not reach a report.
_APPROVED = "approved"

#: The sex value on a bracket curated for everyone.
_ANY_SEX = "any"

# --- age parsing ------------------------------------------------------------
_AGE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)")


def parse_age(raw: str | None) -> float | None:
    """Age in YEARS from free text: '23'→23.0, '6 months'→0.5, '15 days'→~0.041.
    Bare number = years. Unknown unit or non-numeric → None (can't safely bracket)."""
    if not raw:
        return None
    m = _AGE_RE.match(raw)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).lower()
    if unit in ("", "y", "yr", "yrs", "year", "years"):
        return num
    if unit in ("m", "mo", "mon", "mons", "month", "months"):
        return num / 12.0
    if unit in ("w", "wk", "wks", "week", "weeks"):
        return num * 7.0 / 365.25
    if unit in ("d", "day", "days"):
        return num / 365.25
    return None


def canon_sex(raw: str | None) -> str | None:
    """'male' | 'female' from what the report printed, or None when it is not one of the
    two the curation splits on.

    Deliberately stricter than ``normalization._select_gender_range``, which treats
    anything not starting with 'f' as male: here an unrecognised value must not silently
    select the male bracket. Unknown means no sex-specific bracket, which costs a flag; a
    guess costs a wrong one.
    """
    s = (raw or "").strip().lower()
    if s in ("m", "male", "man", "boy"):
        return "male"
    if s in ("f", "female", "woman", "girl"):
        return "female"
    return None


def match_key(name: str | None) -> str:
    """Lookup key for a test name, parameter name, alias, or unit: lower-cased with **all**
    whitespace removed. The curated aliases differ by spacing alone ('hdl/ldl ratio' vs
    'hdl / ldl ratio'), and so do labs' test names — same convention as
    ``extraction._dedupe_results``."""
    return "".join((name or "").lower().split())


# --- lookup + resolution ----------------------------------------------------
class Bracket(NamedTuple):
    """One curated ideal range: whom it is for, and the pair we flag against."""

    age_min: int
    age_max: int
    low: float
    high: float
    sex: str  # 'any' | 'male' | 'female'


@dataclass(frozen=True)
class Lookup:
    """One-shot snapshot of the Spring THP tables for a document's resolution."""

    thp_id_by_name: dict[str, int]  # match_key(name or alias) -> thp id
    approved: dict[int, bool]  # thp id -> doctor-approved AND AI-integrated?
    canonical_name: dict[int, str]  # thp id -> the parameter's own name
    base_unit: dict[int, str]  # thp id -> canon_unit of the parameter's own unit
    #: (thp id, canon_unit of a printed unit) -> (multiplier, offset) into the base unit.
    alt_units: dict[tuple[int, str], tuple[float, float]]
    #: thp id -> [Bracket], unordered.
    age_ranges: dict[int, list[Bracket]]


@dataclass(frozen=True)
class Resolution:
    bounds: tuple[float | None, float | None] | None
    source: str  # "ideal_range" | "report_range"
    matched_parameter: str | None
    matched_group: str | None  # the age bracket used, e.g. "18-60"
    #: None on success; else unmatched | unapproved | no_ideal_range | unit_mismatch
    reason: str | None


def load_lookup(session: Session) -> Lookup:
    """Bulk-read the THP master, its aliases, its alternate units, and its age brackets.

    Exact parameter names are registered before aliases, so a parameter can never be
    shadowed by another's alias. Unapproved parameters are loaded too — that is how we tell
    an unapproved THP from an unknown test.

    Soft-deleted parameters are excluded in SQL rather than filtered afterwards: a deleted
    row must not match a test name at all, not even to report it as unapproved.
    """
    thp_id_by_name: dict[str, int] = {}
    approved: dict[int, bool] = {}
    canonical_name: dict[int, str] = {}
    base_unit: dict[int, str] = {}

    for thp_id, name, units, status, ai_integrated in session.execute(
        select(
            traditional_health_parameters.c.id,
            traditional_health_parameters.c.name,
            traditional_health_parameters.c.units,
            traditional_health_parameters.c.status,
            traditional_health_parameters.c.ai_integrated,
        ).where(traditional_health_parameters.c.deleted_at.is_(None))
    ):
        approved[thp_id] = status == _APPROVED and bool(ai_integrated)
        base_unit[thp_id] = canon_unit(units or "")
        if name:
            canonical_name[thp_id] = name
            thp_id_by_name.setdefault(match_key(name), thp_id)

    # Aliases are how a report's own spelling reaches a parameter at all -- a lab prints
    # "FBS", the catalogue calls it "Fasting Blood Sugar" -- so an unapproved one is not a
    # near-miss to be logged, it is simply not a name for this parameter yet.
    for thp_id, alias in session.execute(
        select(thp_alias.c.thp_id, thp_alias.c.alias).where(
            thp_alias.c.status == _APPROVED, thp_alias.c.thp_id.is_not(None)
        )
    ):
        if alias:
            thp_id_by_name.setdefault(match_key(alias), thp_id)

    alt_units: dict[tuple[int, str], tuple[float, float]] = {}
    for thp_id, name, multiplier, offset in session.execute(
        select(
            thp_alternate_units.c.thp_id,
            thp_alternate_units.c.name,
            thp_alternate_units.c.multiplier,
            thp_alternate_units.c.offset_value,
        )
    ):
        # A non-positive multiplier cannot be inverted into an ordered pair of bounds, so
        # such a row is treated as absent (the test falls back to its report range).
        if name and multiplier and multiplier > 0:
            alt_units[(thp_id, canon_unit(name))] = (multiplier, offset or 0.0)

    age_ranges: dict[int, list[Bracket]] = {}
    for thp_id, age_min, age_max, low, high, sex in session.execute(
        select(
            thp_age_range.c.thp_id,
            thp_age_range.c.age_min,
            thp_age_range.c.age_max,
            _IDEAL_FLOOR,
            _IDEAL_CEILING,
            thp_age_range.c.sex,
        ).where(thp_age_range.c.status == _APPROVED)
    ):
        age_ranges.setdefault(thp_id, []).append(
            Bracket(age_min, age_max, low, high, (sex or _ANY_SEX).strip().lower())
        )

    return Lookup(thp_id_by_name, approved, canonical_name, base_unit, alt_units, age_ranges)


def pick_bracket(
    rows: list[Bracket], age_years: float | None, sex: str | None = None
) -> Bracket | None:
    """The most specific bracket covering this patient, or None.

    Specificity is sex first, then the narrowest age span, ties broken by the lower bound —
    deterministic either way, since the schema forbids neither overlapping ages nor a
    sex-specific row alongside an 'any' one. Sex outranks age width because a parameter is
    curated per sex precisely when the sex is what moves the range (haemoglobin, creatinine,
    ferritin); an 'any' row for such a parameter is the fallback, not the better answer.

    Two ways to get nothing, both deliberate. **No age means no bracket** — a range curated
    for adults must not be applied to an unknown age. And a parameter curated ONLY per sex
    (which is the shape of all 78 sex-specific rows) resolves nothing for a report that did
    not print one, exactly as a printed 'Male: … Female: …' range is left unparsed without a
    gender. Both fall back to the report's own range and are logged for R&D.
    """
    if age_years is None:
        return None
    wanted = canon_sex(sex)
    hits = [
        row
        for row in rows
        if row.age_min <= age_years <= row.age_max and (row.sex == _ANY_SEX or row.sex == wanted)
    ]
    if not hits:
        return None
    return min(
        hits,
        key=lambda row: (row.sex == _ANY_SEX, row.age_max - row.age_min, row.age_min),
    )


def in_report_unit(
    bounds: tuple[float, float], unit: str | None, thp_id: int, lookup: Lookup
) -> tuple[float, float] | None:
    """The ideal range expressed in the unit the report printed, or None if it cannot be.

    The range is curated in the parameter's own unit, so a report printing another unit
    needs converting before the numbers can be compared. Converting the two **bounds** once
    — rather than the value — keeps every downstream comparison (numeric, censored '< 148',
    qualitative) working unchanged on the value exactly as printed. The dashboard defines
    conversion as printed → base (``base = printed * multiplier + offset``), so the inverse
    is applied here; ``load_lookup`` guarantees a positive multiplier, so the pair stays
    ordered.
    """
    printed = canon_unit(unit or "")
    # No printed unit contradicts nothing, and plenty of results (ratios, indices, counts)
    # never print one — assume the parameter's own unit rather than losing the override.
    if not printed or printed == lookup.base_unit.get(thp_id, ""):
        return bounds
    conversion = lookup.alt_units.get((thp_id, printed))
    if conversion is None:
        return None
    multiplier, offset = conversion
    low, high = bounds
    return (low - offset) / multiplier, (high - offset) / multiplier


def resolve(
    test_name: str,
    unit: str | None,
    age_years: float | None,
    lookup: Lookup,
    *,
    sex: str | None = None,
) -> Resolution:
    """Best authoritative range for one test, or a report-range fallback with a reason."""
    thp_id = lookup.thp_id_by_name.get(match_key(test_name))
    if thp_id is None:
        return Resolution(None, "report_range", None, None, "unmatched")
    canonical = lookup.canonical_name.get(thp_id, test_name)
    if not lookup.approved.get(thp_id, False):
        return Resolution(None, "report_range", canonical, None, "unapproved")

    bracket = pick_bracket(lookup.age_ranges.get(thp_id, []), age_years, sex)
    if bracket is None:
        return Resolution(None, "report_range", canonical, None, "no_ideal_range")
    low, high = bracket.low, bracket.high
    # The sex is named in the worklist group only when it narrowed the choice, so an R&D
    # reader can tell "the adult range" from "the adult FEMALE range".
    group = f"{bracket.age_min}-{bracket.age_max}"
    if bracket.sex != _ANY_SEX:
        group = f"{group} {bracket.sex}"

    bounds = in_report_unit((low, high), unit, thp_id, lookup)
    if bounds is None:
        return Resolution(None, "report_range", canonical, group, "unit_mismatch")
    return Resolution(bounds, "ideal_range", canonical, group, None)


if __name__ == "__main__":  # pragma: no cover - self-check
    assert parse_age("23") == 23.0
    assert parse_age("6 months") == 0.5
    assert abs((parse_age("15 days") or 0) - 15 / 365.25) < 1e-9
    assert parse_age("1 year") == 1.0
    assert parse_age("Positive") is None
    assert match_key("HDL / LDL Ratio") == match_key("hdl/ldl ratio") == "hdl/ldlratio"
    assert canon_sex("Male") == canon_sex("m") == "male"
    assert canon_sex("FEMALE") == "female"
    assert canon_sex("Other") is None and canon_sex(None) is None

    # Narrowest covering bracket wins; both ends inclusive; no age -> no bracket.
    _rows = [
        Bracket(0, 150, 1.0, 9.0, "any"),
        Bracket(18, 60, 4.0, 6.0, "any"),
        Bracket(0, 1, 2.0, 3.0, "any"),
    ]
    assert pick_bracket(_rows, 40) == (18, 60, 4.0, 6.0, "any")
    assert pick_bracket(_rows, 0.5) == (0, 1, 2.0, 3.0, "any")
    assert pick_bracket(_rows, 60) == (18, 60, 4.0, 6.0, "any")  # inclusive upper
    assert pick_bracket(_rows, 70) == (0, 150, 1.0, 9.0, "any")
    assert pick_bracket(_rows, None) is None
    assert pick_bracket([], 40) is None

    # Sex outranks a narrower age span, and an unknown sex takes neither half.
    _sexed = [Bracket(0, 120, 13.0, 17.0, "male"), Bracket(0, 120, 12.0, 15.0, "female")]
    assert pick_bracket(_sexed, 40, "F") == (0, 120, 12.0, 15.0, "female")
    assert pick_bracket(_sexed, 40, "male") == (0, 120, 13.0, 17.0, "male")
    assert pick_bracket(_sexed, 40, None) is None
    _mixed = [*_sexed, Bracket(18, 60, 1.0, 2.0, "any")]  # a narrower 'any' still loses
    assert pick_bracket(_mixed, 40, "male") == (0, 120, 13.0, 17.0, "male")

    _lk = Lookup(
        thp_id_by_name={"hemoglobin": 1, "hb": 1, "glucose": 2, "esr": 3},
        approved={1: True, 2: False, 3: True},
        canonical_name={1: "Hemoglobin", 2: "Glucose", 3: "ESR"},
        base_unit={1: "g/dl", 2: "mg/dl", 3: "mm/hr"},
        alt_units={(1, "g/l"): (0.1, 0.0)},  # 130 g/L * 0.1 = 13 g/dL
        age_ranges={
            1: [Bracket(18, 60, 13.0, 17.0, "any")],
            3: [Bracket(0, 150, 0.0, 20.0, "any")],
        },
    )
    assert resolve("Hemoglobin", "g/dL", 40, _lk).bounds == (13.0, 17.0)
    assert resolve("HB", None, 40, _lk).source == "ideal_range"  # alias, no printed unit
    assert resolve("Glucose", "mg/dL", 40, _lk).reason == "unapproved"
    assert resolve("Unknown Test", None, 40, _lk).reason == "unmatched"
    assert resolve("Hemoglobin", "g/dL", 5, _lk).reason == "no_ideal_range"
    assert resolve("Hemoglobin", "g/dL", None, _lk).reason == "no_ideal_range"
    assert resolve("ESR", "mm/hr", 40, _lk).bounds == (0.0, 20.0)
    # A printed unit with a curated conversion: bounds come back in g/L, so the report's
    # own number is compared as printed.
    assert resolve("Hemoglobin", "g/L", 40, _lk).bounds == (130.0, 170.0)
    # A printed unit with no conversion is never guessed at.
    assert resolve("Hemoglobin", "mmol/L", 40, _lk).reason == "unit_mismatch"
    assert resolve("Hemoglobin", "mmol/L", 40, _lk).matched_group == "18-60"
