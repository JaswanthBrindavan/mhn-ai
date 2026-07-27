"""Age parsing, the age-group ladder, and range resolution — pure, no DB."""

from app.services.ideal_ranges import (
    Lookup,
    build_group_ladder,
    has_demographics,
    normalize_gender,
    parse_age,
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


def test_bracket_boundaries_via_ladder():
    # Upper-inclusive brackets; Adult/Older split exclusive/inclusive at 60.
    assert build_group_ladder("0.05", "M")[0] == "neonate male"  # ~18 days
    assert build_group_ladder("1", "M")[0] == "infant male"  # 1.0 yr -> Infant
    assert build_group_ladder("3", "M")[0] == "toddler male"  # 3.0 -> Toddler
    assert build_group_ladder("10", "M")[0] == "child male"
    assert build_group_ladder("18", "M")[0] == "adolescent male"  # 18.0 -> Adolescent
    assert build_group_ladder("18.5", "M")[0] == "adult male"
    assert build_group_ladder("59.9", "M")[0] == "adult male"
    assert build_group_ladder("60", "M")[0] == "older male"  # 60.0 -> Older


def test_normalize_gender():
    assert normalize_gender("M") == "Male"
    assert normalize_gender("male") == "Male"
    assert normalize_gender("F") == "Female"
    assert normalize_gender("Female") == "Female"
    assert normalize_gender("x") is None
    assert normalize_gender(None) is None


def test_group_ladder_shapes():
    assert build_group_ladder("40", "M") == ["adult male", "adult all", "all male", "all"]
    # Older falls through the adult tiers.
    assert build_group_ladder("70", "F") == [
        "older female",
        "older all",
        "adult female",
        "adult all",
        "all female",
        "all",
    ]
    # Pediatric brackets splice the Children aggregate.
    assert build_group_ladder("5", "M") == [
        "child male",
        "child all",
        "children male",
        "children all",
        "all male",
        "all",
    ]
    # Missing gender drops gender-specific tiers.
    assert build_group_ladder("40", None) == ["adult all", "all"]
    # Missing age drops bracket tiers, keeps gender + all.
    assert build_group_ladder(None, "F") == ["all female", "all"]
    # Neither still tries the demographic-agnostic "all".
    assert build_group_ladder(None, None) == ["all"]


def test_has_demographics():
    assert has_demographics("40", "M") is True
    assert has_demographics("40", None) is True
    assert has_demographics(None, "F") is True
    assert has_demographics(None, None) is False
    assert has_demographics("junk", "junk") is False


def _lookup() -> Lookup:
    return Lookup(
        pkid_by_name={"hemoglobin": 1, "hb": 1, "glucose": 2, "esr": 3},
        approved={1: True, 2: False, 3: True},
        canonical_name={1: "Hemoglobin", 2: "Glucose", 3: "ESR"},
        ideal={(1, "adult male"): (13.0, 17.0), (3, "all"): (0.0, 20.0)},
    )


def test_resolve_exact_and_alias_hit():
    lk = _lookup()
    ladder = build_group_ladder("40", "M")
    exact = resolve("Hemoglobin", lk, ladder)
    assert exact.bounds == (13.0, 17.0)
    assert exact.source == "ideal_range"
    assert exact.matched_parameter == "Hemoglobin"
    assert exact.matched_group == "adult male"
    assert resolve("HB", lk, ladder).bounds == (13.0, 17.0)  # alias resolves to same


def test_resolve_unapproved_and_unmatched():
    lk = _lookup()
    ladder = build_group_ladder("40", "M")
    assert resolve("Glucose", lk, ladder).reason == "unapproved"
    assert resolve("Glucose", lk, ladder).bounds is None
    assert resolve("Unknown Test", lk, ladder).reason == "unmatched"
    assert resolve("Unknown Test", lk, ladder).matched_parameter is None


def test_resolve_no_range_for_group_falls_back():
    lk = _lookup()
    # Hemoglobin only has an "adult male" range; a female child finds none.
    res = resolve("Hemoglobin", lk, build_group_ladder("5", "F"))
    assert res.bounds is None
    assert res.reason == "no_ideal_range"


def test_resolve_uses_demographic_agnostic_all_range():
    lk = _lookup()
    # ESR only has an "all" range -> matches for any ladder.
    assert resolve("ESR", lk, build_group_ladder("40", "M")).bounds == (0.0, 20.0)
    assert resolve("ESR", lk, build_group_ladder(None, None)).bounds == (0.0, 20.0)
