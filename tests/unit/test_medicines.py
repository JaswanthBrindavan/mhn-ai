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


def test_an_abbreviations_slots_are_marked_as_this_modules_choice() -> None:
    """BD says twice a day. Morning and night is what this module picked, not what the
    prescription said — and stored as four floats it is indistinguishable from a printed
    "1-0-1" unless something says so."""
    assert norm("BD")["schedule_inferred"] is True
    assert norm("twice a day")["schedule_inferred"] is True
    assert norm("1-0-1")["schedule_inferred"] is False
    # HS is not a rate: "at bedtime" is a time of day the document actually stated.
    assert norm("HS")["schedule_inferred"] is False
    assert norm("1 tab OD at night")["schedule_inferred"] is False


def test_interval_notation_is_left_null_rather_than_guessed() -> None:
    """Q6H states a rate this shape has no field for. Null is the honest answer; the
    module docstring names it as the first gap to close."""
    assert normalize_frequency("Q6H") is None
    assert normalize_frequency("q8h") is None
    assert normalize_frequency("every 6 hours") is None


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
    """ "PERSOL AC 2.5 GEL" is a benzoyl peroxide gel, not a dose before food.

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
    """ "0.75 MG MORNING" restates the strength: one dose in the morning, not 0.75 of one."""
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


@pytest.mark.parametrize(
    "text",
    [
        "Tab Methotrexate 1-0-0 once a week",
        "1-0-0 weekly",
        "1 tablet weekly",
        "1-0-0 once a month",
        "1-0-0 monthly",
        "1-0-0 alternate day",
        "1-0-0 on alternate days",
        "1-0-0 every other day",
        "1-0-0 QOD",
        "1-0-0 every 3 days",
    ],
)
def test_a_non_daily_period_refuses_rather_than_reporting_a_daily_dose(text: str) -> None:
    """The rule the module already applied to a bare "Alternate day", now held when a dose
    matrix is printed beside the period — which is how a weekly medicine is written.

    The returned shape counts doses per day and has no field for a period, so there is no
    way to say "one, on one day in seven". Reading it as a daily schedule multiplies the
    dose by seven, and nothing downstream can see that it happened: weekly methotrexate
    taken daily is the standard example of why. Null plus the raw text is the only honest
    answer this shape can give.
    """
    assert normalize_frequency(text) is None


def test_a_course_that_changes_refuses_rather_than_taking_the_first_half() -> None:
    """A taper prints two schedules. Which applies depends on the day, which this shape
    cannot say either — and silently keeping the first drops the rest of the course."""
    assert normalize_frequency("1-1-1 for 3 days then 1-0-1") is None
    # One schedule stated twice is not a taper.
    assert slots("1-0-1, repeat 1-0-1") == (1.0, 0.0, 0.0, 1.0)


def test_a_duration_in_days_is_not_mistaken_for_a_period() -> None:
    """ "for 5 days" is how long the course runs, not how often a dose is taken."""
    assert slots("1-0-1 for 5 days") == (1.0, 0.0, 0.0, 1.0)
    assert slots("1-0-1 x 10 days after food") == (1.0, 0.0, 0.0, 1.0)


def test_stat_is_not_a_morning_dose() -> None:
    """STAT is one dose, immediately: a point in time, not a time of day and not a repeat.

    It used to map to the morning slot, which turned a dose given the moment it was
    written into an instruction to take one tomorrow morning.
    """
    assert normalize_frequency("STAT") is None
    assert normalize_frequency("Inj. Monocef 1gm STAT") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 tab OD at night", (0.0, 0.0, 0.0, 1.0)),
        ("OD at bedtime", (0.0, 0.0, 0.0, 1.0)),
        ("Once daily at night", (0.0, 0.0, 0.0, 1.0)),
        ("1 tab OD in the evening", (0.0, 0.0, 1.0, 0.0)),
    ],
)
def test_a_printed_time_of_day_beats_the_abbreviations_default(
    text: str, expected: tuple[float, ...]
) -> None:
    """OD says how MANY doses; it does not say when. When the document also says when, the
    document wins.

    It did not: "at night" named no slot the parser could read — the phrase needed a "the"
    — so the schedule fell through to OD's default of morning, and the one thing the
    document actually stated about timing was the thing that got dropped. Once daily at
    night is how statins and PPIs are routinely written.
    """
    assert slots(text) == expected


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
        # Genuine equivalents, folded in rather than given values of their own.
        ("Suspension", "Syrup"),  # dosed and taken exactly as a syrup is
        ("Vial", "Injection"),
        ("IV infusion", "Injection"),
        # Indian short forms, none of them guessable. Eye and nasal drops are both Drops:
        # the site is not part of the vocabulary.
        ("E/D", "Drops"),
        ("N/D", "Drops"),
        # An ointment folds into Cream: both go on the skin, so the base (greasy vs
        # aqueous) is all that is lost, and that is display detail, not a route.
        ("E/O", "Cream"),
        ("Oint.", "Cream"),
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
    # Sachet/Granules/Powder joined this list when the vocabulary became Spring's eight:
    # a sachet of ORS is not a tablet, a capsule or a syrup, so it reports no form.
    # "Patch" is here too — it IS one of the eight, but no document writes it as a printed
    # abbreviation, so the lookup table has no entry and the model classifies it instead.
    [
        "GEL",
        "Lotion",
        "Nasal Spray",
        "Suppository",
        "Patch",
        "Gargle",
        "Solution",
        "Sachet",
        "Granules",
        "Powder",
    ],
)
def test_a_form_outside_the_vocabulary_is_null_not_the_nearest_box(text: str) -> None:
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
    produce a value outside it — including one added later by hand.

    The list itself is Spring's, so what it must EQUAL is asserted against their own
    migration in ``test_dosage_form_vocabulary.py`` rather than restated here — a second
    hand-written copy is the drift this vocabulary already suffered once.
    """
    from app.services.medicines import _FORM_SYNONYMS

    assert set(_FORM_SYNONYMS.values()) <= DOSAGE_FORMS


def test_parse_dose_forms() -> None:
    assert parse_dose("1/2") == 0.5
    assert parse_dose("½") == 0.5
    assert parse_dose("0.5") == 0.5
    assert parse_dose("half") == 0.5
    assert parse_dose("2") == 2.0
    assert parse_dose("banana") is None
