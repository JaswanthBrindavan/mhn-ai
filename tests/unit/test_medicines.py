"""Dosing notation -> daily schedule.

Every case here is a notation that appeared on a real Indian prescription. The parser is
deterministic, so these are exact-value assertions rather than smoke tests — the point of
doing this in Python instead of asking the model is that the answer cannot drift.
"""

from typing import Any

import pytest

from app.services.medicines import (
    DOSAGE_FORMS,
    normalize_form,
    normalize_frequency,
    parse_dose,
)


def norm(text: str) -> dict[str, Any]:
    result = normalize_frequency(text)
    assert result is not None, f"{text!r} should have normalised"
    return result


def slots(text: str) -> tuple[float, float, float, float]:
    result = norm(text)
    return (result["morning"], result["afternoon"], result["evening"], result["night"])


# --- the dose matrix --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1-0-1", (1.0, 0.0, 0.0, 1.0)),
        ("1-1-1", (1.0, 1.0, 0.0, 1.0)),
        ("1-1-1-1", (1.0, 1.0, 1.0, 1.0)),
        ("0-1-0-0", (0.0, 1.0, 0.0, 0.0)),
        # Fractions, in every form a document writes them.
        ("1/2-0-1/2", (0.5, 0.0, 0.0, 0.5)),
        ("0.5-0-0.5", (0.5, 0.0, 0.0, 0.5)),
        ("½-0-½", (0.5, 0.0, 0.0, 0.5)),
        # Spaces inside the fraction: what a PDF text layer actually yields.
        ("Twice Daily ( 1 / 2 - 0 - 0 - 1 / 2 )", (0.5, 0.0, 0.0, 0.5)),
        # A ruled line between slots rather than one hyphen.
        ("1----0---1", (1.0, 0.0, 0.0, 1.0)),
    ],
)
def test_dose_matrix(text: str, expected: tuple[float, ...]) -> None:
    assert slots(text) == expected


def test_a_four_slot_matrix_is_not_squashed_into_three() -> None:
    """The slot count decides the time of day, so it has to survive.

    A four-slot 0-0-1-0 is an evening dose; read as three slots it becomes night. Nothing
    downstream could detect that, which is why it is asserted rather than assumed.
    """
    assert slots("0-0-1-0") == (0.0, 0.0, 1.0, 0.0)
    assert slots("0-0-1") == (0.0, 0.0, 0.0, 1.0)


def test_a_date_is_not_read_as_a_dosage() -> None:
    # "12-03-25" is a perfect dose matrix by shape; only the magnitude gives it away.
    assert normalize_frequency("Start 12-03-25") is None


# --- Latin ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("OD", (1.0, 0.0, 0.0, 0.0)),
        ("BD", (1.0, 0.0, 0.0, 1.0)),
        ("TDS", (1.0, 1.0, 0.0, 1.0)),
        ("QID", (1.0, 1.0, 1.0, 1.0)),
        ("HS", (0.0, 0.0, 0.0, 1.0)),
        ("b.d.", (1.0, 0.0, 0.0, 1.0)),
    ],
)
def test_latin_abbreviations(text: str, expected: tuple[float, ...]) -> None:
    assert slots(text) == expected


def test_food_modifiers() -> None:
    assert norm("1-0-1 AC")["with_food"] is False
    assert norm("1-0-1 PC")["with_food"] is True
    assert norm("1-0-1 after food")["with_food"] is True
    assert norm("1-0-1 before food")["with_food"] is False
    assert norm("1-0-1")["with_food"] is None


def test_as_needed() -> None:
    assert norm("SOS")["as_needed"] is True
    assert norm("take when required")["as_needed"] is True
    assert norm("1-0-1")["as_needed"] is False


def test_ac_in_a_brand_name_is_not_read_as_a_food_modifier() -> None:
    """"PERSOL AC 2.5 GEL" is a benzoyl peroxide gel, not a dose before food.

    AC and PC are two letters and collide with brand names, so they only count beside a
    real schedule. Without this the name is also truncated at "PERSOL".
    """
    assert normalize_frequency("PERSOL AC 2.5 GEL 30 GM") is None


# --- English ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("twice a day", (1.0, 0.0, 0.0, 1.0)),
        ("Once Daily", (1.0, 0.0, 0.0, 0.0)),
        ("thrice a day", (1.0, 1.0, 0.0, 1.0)),
        ("1 tablet in the morning", (1.0, 0.0, 0.0, 0.0)),
        ("Morning-1, Night-1", (1.0, 0.0, 0.0, 1.0)),
        # The hour picks the slot it falls in: evening runs 17:00-20:59, so 8 PM is an
        # evening dose and night does not begin until 21:00.
        ("8 AM and 8 PM", (1.0, 0.0, 1.0, 0.0)),
        ("8 AM and 10 PM", (1.0, 0.0, 0.0, 1.0)),
        ("AT 6 AM, 2PM, 10 PM", (1.0, 1.0, 0.0, 1.0)),
    ],
)
def test_english_forms(text: str, expected: tuple[float, ...]) -> None:
    assert slots(text) == expected


def test_a_strength_before_a_slot_is_one_dose_not_a_count() -> None:
    """"0.75 MG MORNING" restates the strength: one dose in the morning, not 0.75 of one."""
    assert slots("0.75 MG MORNING") == (1.0, 0.0, 0.0, 0.0)


# --- refusing to guess ------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [None, "", "   ", "as directed", "Alternate day", "apply locally", "Qty: 90"],
)
def test_unparseable_stays_null(text: str | None) -> None:
    """A schedule is never invented — a wrong one is worse than none.

    "Alternate day" is deliberate: the schedule counts doses per *day*, and an
    alternate-day medicine takes none on half of them. Any daily figure would overstate
    the dose, so the phrase stays in frequency_raw and the schedule stays null.
    """
    assert normalize_frequency(text) is None


# --- dosage form ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The same form, written the four ways one corpus contained.
        ("Tab.", "Tablet"),
        ("TAB", "Tablet"),
        ("TABLET", "Tablet"),
        ("Tablets", "Tablet"),
        ("Cap", "Capsule"),
        ("Syp", "Syrup"),
        ("INJ", "Injection"),
        # Genuine equivalents, folded into the nine rather than given values of their own.
        ("Suspension", "Syrup"),  # dosed and taken exactly as a syrup is
        ("Vial", "Injection"),
        ("IV infusion", "Injection"),
        ("Sachet", "Powder"),
        ("Granules", "Powder"),
        # Indian short forms, none of them guessable. Eye and nasal drops are both Drops:
        # the site is not part of the vocabulary.
        ("E/D", "Drops"),
        ("N/D", "Drops"),
        ("E/O", "Ointment"),
        ("Rotacap", "Inhaler"),  # a dry-powder inhaler, not a capsule to swallow
        ("Nebuliser", "Inhaler"),
        # Taken from the name when there is no separate column for it.
        ("Tab. DOLO 650", "Tablet"),
        ("DAPAGLIFLOZIN-TABLET-5MG-DAPEFY", "Tablet"),
        ("Inj. Monocef 1gm", "Injection"),
    ],
)
def test_dosage_forms(text: str, expected: str) -> None:
    assert normalize_form(text) == expected


@pytest.mark.parametrize(
    "text",
    ["GEL", "Lotion", "Nasal Spray", "Suppository", "Patch", "Gargle", "Solution"],
)
def test_a_form_outside_the_nine_is_null_not_the_nearest_box(text: str) -> None:
    """These are real forms with no honest home in the vocabulary, so they stay null.

    Route is clinical. A suppository reported as a Tablet is a swallowing instruction for
    something that must not be swallowed — a guess here is worse than an admission of
    ignorance, and ``form_raw`` still carries what the document printed.
    """
    assert normalize_form(text) is None


@pytest.mark.parametrize("text", [None, "", "650mg", "wibble", "1-0-1"])
def test_a_non_form_is_null(text: str | None) -> None:
    assert normalize_form(text) is None


def test_nothing_escapes_the_published_vocabulary() -> None:
    """``DOSAGE_FORMS`` is the contract every consumer switches on, so no mapping may
    produce a value outside it — including one added later by hand."""
    from app.services.medicines import _FORM_SYNONYMS

    assert set(_FORM_SYNONYMS.values()) <= DOSAGE_FORMS
    assert {
        "Tablet", "Capsule", "Syrup", "Injection", "Drops",
        "Cream", "Ointment", "Inhaler", "Powder",
    } == DOSAGE_FORMS


def test_parse_dose_forms() -> None:
    assert parse_dose("1/2") == 0.5
    assert parse_dose("½") == 0.5
    assert parse_dose("0.5") == 0.5
    assert parse_dose("half") == 0.5
    assert parse_dose("2") == 2.0
    assert parse_dose("banana") is None
