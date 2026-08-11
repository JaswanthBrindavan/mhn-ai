"""When the next dose is due, and who is allowed to work it out.

``vaccinations.next_due_on`` is the only real Spring column any section extraction writes,
and Spring's reminder index reads it. So whatever decides that value decides when a parent
is told to bring a child back for a vaccine.

It used to be the model. The prompt listed "due after 4 weeks" among the WINDOWS to return
the start of, which is not a window at all — returning a date for it means computing
date_given + 28 days. Probed against a real-shaped immunisation card:

    Date given: 04/03/2026
    Next dose due after 4 weeks
      -> next_due_date "01/04/2026"

That date is printed nowhere in the document. It was right that time, is unverifiable in
general, and was stored indistinguishably from a date the certificate actually carried.

Now the model transcribes the phrase into ``next_due_interval`` and this arithmetic
happens in Python, where month lengths and leap years can be tested.
"""

import pytest

from app.services.classification import DocumentSection
from app.services.dates import add_interval
from app.services.section_extraction import build_payload
from app.services.section_specs import VaccinationFields, spec_for

VACCINATIONS = spec_for(DocumentSection.VACCINATIONS)


def _fields(**overrides: object) -> dict:
    base = {"vaccine_name": "Pentavalent", "date_given": "04/03/2026"}
    base.update(overrides)
    return build_payload(VACCINATIONS, VaccinationFields(**base))["fields"]  # type: ignore[arg-type]


# --- the arithmetic ---------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "phrase", "expected"),
    [
        ("2026-03-04", "after 4 weeks", "2026-04-01"),
        ("2026-03-04", "28 days", "2026-04-01"),
        ("2026-03-04", "next dose in 6 months", "2026-09-04"),
        ("2026-03-04", "after one month", "2026-04-04"),
        # A range gives the day the dose FIRST becomes due, like a printed window.
        ("2026-03-04", "6-8 weeks", "2026-04-15"),
        ("2026-03-04", "4 to 6 weeks", "2026-04-01"),
        # Month arithmetic clamps rather than overflowing into the next month.
        ("2026-01-31", "1 month", "2026-02-28"),
        ("2024-02-29", "1 year", "2025-02-28"),
    ],
)
def test_intervals_are_computed_in_python(start: str, phrase: str, expected: str) -> None:
    assert add_interval(start, phrase) == expected


@pytest.mark.parametrize(
    ("start", "phrase"),
    [
        ("2026-03-04", "when convenient"),  # not an interval
        ("2026-03-04", "after 4 fortnights"),  # unit we do not know
        ("2026-03-04", "0 weeks"),  # not an interval either
        (None, "4 weeks"),  # nothing to add to
        ("2026-03-04", None),
        ("not a date", "4 weeks"),
    ],
)
def test_an_unreadable_interval_refuses_rather_than_guessing(start, phrase) -> None:
    """The same rule the rest of this codebase follows. A missing reminder is recoverable;
    a reminder on the wrong day is not."""
    assert add_interval(start, phrase) is None


# --- the payload ------------------------------------------------------------


def test_a_stated_interval_fills_the_date_the_model_left_null() -> None:
    fields = _fields(next_due_interval="due after 4 weeks")

    assert fields["next_due_date"] == "2026-04-01"
    # The phrase is kept, which is what makes the date's provenance readable: a
    # next_due_date beside a non-null interval was derived, one beside a null interval
    # was printed on the document.
    assert fields["next_due_interval"] == "due after 4 weeks"


def test_a_printed_date_always_wins_over_a_derived_one() -> None:
    """If the record printed a date, that is the fact; the interval is only a fallback."""
    fields = _fields(next_due_date="20/04/2026", next_due_interval="after 4 weeks")

    assert fields["next_due_date"] == "2026-04-20"


def test_no_interval_and_no_date_stays_null() -> None:
    """A completed series states neither, and nothing should be invented for it."""
    assert _fields()["next_due_date"] is None
