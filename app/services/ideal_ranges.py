"""Resolve the doctor-approved ideal range for a lab result, by age group.

For each extracted test we try to match it to an R&D-approved THP (health parameter)
and pick the ideal range for the patient's age-group ladder; that range then overrides
the report's printed reference range for the abnormal-flag calculation. A miss (test not
a known parameter, parameter not approved, or no ideal range for the group) leaves the
report's own range in charge and is recorded as an R&D worklist fallback.

All logic here is deterministic Python. The Spring-owned parameter tables are read once
per document (``load_lookup``); everything else is pure and unit-testable without a DB.

Ported/adapted from the reference implementation
``D:\\mhn-ai-main-1\\mhn-ai-main\\source\\utils.py`` (match_parameter / load_ideal_values)
and the age-group ladder in its ``docs/main-backend-context.md`` §3.

# ponytail: curated cutoffs + first-hit ladder, not a demographics engine. The Spring
# table names + approval predicate are UNCONFIRMED (feature is flag-gated off) — the two
# marked constants below are the only things to change when they are.
"""

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.spring import parameter_aliases, parameter_ideal_values, parameters

# --- approval predicate (TO CONFIRM with Spring/R&D) ------------------------
#: A THP is usable only when doctor-approved. Reference marks approval with a status
#: string plus a non-null approver. Both are guesses until the Spring schema is final.
_APPROVED_STATUS = "approved"


def _is_approved(status: str | None, approved_by_id: int | None) -> bool:
    return (status or "").strip().lower() == _APPROVED_STATUS and approved_by_id is not None


# --- age parsing + brackets -------------------------------------------------
_AGE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)")

#: Ordered, upper-inclusive age-group cutoffs in YEARS. First hit wins. Keep in sync with
#: docs/main-backend-context.md §3 (Neonate ≤28d, Infant ≤1y, Toddler ≤3y, Child ≤10y,
#: Adolescent ≤18y). Adult/Older split at 60 is exclusive/inclusive, handled below.
_NEONATE_MAX = 28 / 365.25
_ADULT_MAX_EXCLUSIVE = 60.0
_AGE_GROUP_CUTOFFS: list[tuple[float, str]] = [
    (_NEONATE_MAX, "Neonate"),
    (1.0, "Infant"),
    (3.0, "Toddler"),
    (10.0, "Child"),
    (18.0, "Adolescent"),
]
_PEDIATRIC = frozenset({"Neonate", "Infant", "Toddler", "Child", "Adolescent"})


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


def normalize_gender(raw: str | None) -> str | None:
    """'M'/'male'→'Male', 'F'/'female'→'Female', anything else → None."""
    if not raw:
        return None
    g = raw.strip().lower()
    if g in ("m", "male"):
        return "Male"
    if g in ("f", "female"):
        return "Female"
    return None


def _bracket(age_years: float) -> str:
    for cutoff, name in _AGE_GROUP_CUTOFFS:
        if age_years <= cutoff:
            return name
    return "Adult" if age_years < _ADULT_MAX_EXCLUSIVE else "Older"


def has_demographics(age_raw: str | None, gender_raw: str | None) -> bool:
    """True when the report gave us something to bracket on. Drives fallback-log noise:
    a miss with no demographics is the report's gap, not R&D's."""
    return parse_age(age_raw) is not None or normalize_gender(gender_raw) is not None


def build_group_ladder(age_raw: str | None, gender_raw: str | None) -> list[str]:
    """First-hit-wins list of lowercased group keys to try, most specific first.

    e.g. Adult Male → 'adult male','adult all','all male','all'. Older falls through
    Adult; pediatric brackets splice a 'Children' aggregate; a missing gender drops the
    gender-specific tiers; a missing age drops the bracket tiers. Always ends with 'all'
    (the demographic-agnostic range applies to everyone)."""
    age = parse_age(age_raw)
    gender = normalize_gender(gender_raw)
    bracket = _bracket(age) if age is not None else None

    chain: list[tuple[str, str]] = []

    def add(b: str) -> None:
        if gender is not None:
            chain.append((b, gender))
        chain.append((b, "All"))

    if bracket is not None:
        add(bracket)
        if bracket == "Older":
            add("Adult")  # older adults fall through to the adult range
        elif bracket in _PEDIATRIC:
            add("Children")  # pediatric aggregate
    add("All")

    groups: list[str] = []
    seen: set[str] = set()
    for b, g in chain:
        key = "all" if (b == "All" and g == "All") else f"{b} {g}".lower()
        if key not in seen:
            seen.add(key)
            groups.append(key)
    return groups


# --- lookup + resolution ----------------------------------------------------
@dataclass(frozen=True)
class Lookup:
    """One-shot snapshot of the Spring THP tables for a document's resolution."""

    pkid_by_name: dict[str, int]  # lowercased name/alias -> parameter pkid
    approved: dict[int, bool]  # pkid -> doctor-approved?
    canonical_name: dict[int, str]  # pkid -> the parameter's own name
    ideal: dict[tuple[int, str], tuple[float | None, float | None]]  # (pkid, group) -> min,max


@dataclass(frozen=True)
class Resolution:
    bounds: tuple[float | None, float | None] | None
    source: str  # "ideal_range" | "report_range"
    matched_parameter: str | None
    matched_group: str | None
    reason: str | None  # None on success; else "unmatched"|"unapproved"|"no_ideal_range"


def load_lookup(session: Session) -> Lookup:
    """Bulk-read the parameter master, aliases, and ideal values. Exact names win over
    aliases (matching the reference); all parameters loaded so we can tell an unapproved
    THP from an unknown one."""
    pkid_by_name: dict[str, int] = {}
    approved: dict[int, bool] = {}
    canonical_name: dict[int, str] = {}

    for pkid, name, status, approved_by_id in session.execute(
        select(
            parameters.c.pkid, parameters.c.name, parameters.c.status, parameters.c.approved_by_id
        )
    ):
        approved[pkid] = _is_approved(status, approved_by_id)
        if name:
            canonical_name[pkid] = name
            pkid_by_name.setdefault(name.strip().lower(), pkid)

    for pkid, alias in session.execute(
        select(parameter_aliases.c.parameter_id, parameter_aliases.c.alias)
    ):
        if alias and pkid in approved:
            pkid_by_name.setdefault(alias.strip().lower(), pkid)

    ideal: dict[tuple[int, str], tuple[float | None, float | None]] = {}
    for pkid, group, lo, hi in session.execute(
        select(
            parameter_ideal_values.c.parameter_id,
            parameter_ideal_values.c.group,
            parameter_ideal_values.c.ideal_value_min,
            parameter_ideal_values.c.ideal_value_max,
        )
    ):
        if group is not None:
            ideal[(pkid, group.strip().lower())] = (lo, hi)

    return Lookup(pkid_by_name, approved, canonical_name, ideal)


def resolve(test_name: str, lookup: Lookup, ladder: list[str]) -> Resolution:
    """Best authoritative range for one test, or a report-range fallback with a reason."""
    pkid = lookup.pkid_by_name.get((test_name or "").strip().lower())
    if pkid is None:
        return Resolution(None, "report_range", None, None, "unmatched")
    canonical = lookup.canonical_name.get(pkid, test_name)
    if not lookup.approved.get(pkid, False):
        return Resolution(None, "report_range", canonical, None, "unapproved")
    for group in ladder:
        rng = lookup.ideal.get((pkid, group))
        if rng is not None and rng[0] is not None and rng[1] is not None:
            return Resolution(rng, "ideal_range", canonical, group, None)
    return Resolution(None, "report_range", canonical, None, "no_ideal_range")


if __name__ == "__main__":  # pragma: no cover - self-check
    assert parse_age("23") == 23.0
    assert parse_age("6 months") == 0.5
    assert abs((parse_age("15 days") or 0) - 15 / 365.25) < 1e-9
    assert parse_age("1 year") == 1.0
    assert parse_age("Positive") is None
    assert _bracket(0.05) == "Neonate"
    assert _bracket(1.0) == "Infant"
    assert _bracket(3.0) == "Toddler"
    assert _bracket(18.0) == "Adolescent"
    assert _bracket(59.9) == "Adult"
    assert _bracket(60.0) == "Older"
    assert normalize_gender("M") == "Male"
    assert normalize_gender("female") == "Female"
    assert normalize_gender("?") is None
    assert build_group_ladder("40", "M") == ["adult male", "adult all", "all male", "all"]
    assert build_group_ladder("70", "F") == [
        "older female",
        "older all",
        "adult female",
        "adult all",
        "all female",
        "all",
    ]
    assert build_group_ladder("5", "M") == [
        "child male",
        "child all",
        "children male",
        "children all",
        "all male",
        "all",
    ]
    assert build_group_ladder("40", None) == ["adult all", "all"]
    assert build_group_ladder(None, None) == ["all"]

    _lk = Lookup(
        pkid_by_name={"hemoglobin": 1, "hb": 1, "glucose": 2, "esr": 3},
        approved={1: True, 2: False, 3: True},
        canonical_name={1: "Hemoglobin", 2: "Glucose", 3: "ESR"},
        ideal={(1, "adult male"): (13.0, 17.0), (3, "all"): (0.0, 20.0)},
    )
    adult_male = build_group_ladder("40", "M")
    assert resolve("Hemoglobin", _lk, adult_male).bounds == (13.0, 17.0)
    assert resolve("HB", _lk, adult_male).source == "ideal_range"  # alias
    assert resolve("Glucose", _lk, adult_male).reason == "unapproved"
    assert resolve("Unknown Test", _lk, adult_male).reason == "unmatched"
    assert resolve("Hemoglobin", _lk, build_group_ladder("5", "F")).reason == "no_ideal_range"
    assert resolve("ESR", _lk, adult_male).bounds == (0.0, 20.0)  # only an 'all' range
