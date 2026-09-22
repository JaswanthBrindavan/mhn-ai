"""A vaccination document is a schedule, not a record.

``VaccinationFields`` used to be a single flat record — one vaccine, one dose, one date
given, one next due date. A real immunisation history is two tables: the doses already
administered, and the doses still to come. Measured on a nine-row specimen, what reached
the app was:

    Hepatitis B, Dose 1, 14-Aug-2026, no Next due.

Four administered doses dropped, and every upcoming dose lost outright — the next one due
belonged to a different vaccine in the other table, so the flat field had nowhere to put
it. Nothing reached ``vaccinations.next_due_on``, so the person was never reminded.

``doses`` now carries every row with its own dates, and the flat ``next_due_date`` — the
one written to ``next_due_on``, the one a reminder is raised from — is CHOSEN from that
list in code: earliest still ahead, else the latest already past. The tests below pin the
choice, because it decides when somebody is told to come back for a vaccine.

A single-dose certificate keeps a one-row or empty list and is untouched by all of it,
which is the other thing pinned here.
"""

from datetime import date

import pytest

from app.services.classification import DocumentSection
from app.services.section_extraction import build_payload
from app.services.section_specs import VaccinationFields, spec_for

VACCINATIONS = spec_for(DocumentSection.VACCINATIONS)

#: The day the specimen was read on a device. Fixed so "still ahead" means something.
TODAY = date(2026, 9, 22)


def _payload(**overrides: object) -> dict:
    base: dict = {"vaccine_name": "Hepatitis B", "dose_info": "Dose 1", "doses": []}
    base.update(overrides)
    return build_payload(VACCINATIONS, VaccinationFields(**base), today=TODAY)  # type: ignore[arg-type]


def _fields(**overrides: object) -> dict:
    return _payload(**overrides)["fields"]


def _codes(payload: dict) -> list[str]:
    return [flag["code"] for flag in payload["flags"]]


def _given(name: str, dose: str, when: str, **extra: object) -> dict:
    return {"vaccine_name": name, "dose_info": dose, "date_given": when, **extra}


def _due(name: str, dose: str, when: str) -> dict:
    return {"vaccine_name": name, "dose_info": dose, "next_due_date": when}


#: The specimen, exactly as printed: five administered doses and four upcoming ones.
SPECIMEN = [
    _given("Hepatitis B", "Dose 1", "14/08/2026", batch_lot="HB-7781", site="Left deltoid"),
    _given("Hepatitis B", "Dose 2", "12/09/2026", batch_lot="HB-7902", site="Left deltoid"),
    _given("Influenza QIV", "Annual", "12/09/2026", vaccinator="Dr S. Rao"),
    _given("Tdap", "Booster", "02/06/2026"),
    _given("Typhoid Vi-PS", "Single", "20/03/2026"),
    _due("Rabies (pre-exposure)", "Dose 2 of 3", "27/09/2026"),
    _due("Hepatitis A", "Dose 1", "26/10/2026"),
    _due("Hepatitis B", "Dose 3", "12/03/2027"),
    _due("Influenza", "Annual", "12/09/2027"),
]


# --- every dose survives ----------------------------------------------------


def test_every_dose_in_the_document_is_kept() -> None:
    """The defect itself: four administered doses and four upcoming ones were dropped."""
    doses = _fields(date_given="14/08/2026", doses=SPECIMEN)["doses"]

    assert len(doses) == 9
    assert [d["vaccine_name"] for d in doses[:5]] == [
        "Hepatitis B",
        "Hepatitis B",
        "Influenza QIV",
        "Tdap",
        "Typhoid Vi-PS",
    ]
    # The columns the history table prints beside each dose come with it.
    assert doses[0]["batch_lot"] == "HB-7781"
    assert doses[0]["site"] == "Left deltoid"
    assert doses[2]["vaccinator"] == "Dr S. Rao"


def test_a_dose_keeps_its_own_dates_normalised() -> None:
    """Each row's dates are ISO'd like the flat ones, not left as the model typed them."""
    doses = _fields(doses=SPECIMEN)["doses"]

    assert doses[0]["date_given"] == "2026-08-14"
    assert doses[0]["next_due_date"] is None
    # An upcoming dose is marked by having no date given. That is the only marker.
    assert doses[5]["date_given"] is None
    assert doses[5]["next_due_date"] == "2026-09-27"


# --- which dose drives the reminder -----------------------------------------


def test_the_reminder_is_the_earliest_dose_still_ahead() -> None:
    """The specimen lists four upcoming doses; 27-Sep is the errand, 12-Sep-2027 is not."""
    fields = _fields(date_given="14/08/2026", doses=SPECIMEN)

    assert fields["next_due_date"] == "2026-09-27"


def test_a_date_already_past_is_not_preferred_to_one_ahead() -> None:
    """A missed dose does not become "next" just by being the smallest number."""
    doses = [_due("Tdap", "Booster", "01/01/2026"), _due("Rabies", "Dose 2", "27/09/2026")]

    assert _fields(doses=doses)["next_due_date"] == "2026-09-27"


def test_when_every_dose_is_past_the_latest_one_is_the_outstanding_one() -> None:
    """Not null — a missed booster is exactly what an upcoming screen is for. And not the
    earliest: of several missed doses the last one supersedes the ones before it."""
    doses = [
        _given("Typhoid Vi-PS", "Single", "10/01/2026"),
        _due("Hepatitis A", "Dose 2", "20/03/2026"),
        _due("Tdap", "Booster", "05/05/2026"),
    ]

    assert _fields(date_given="10/01/2026", doses=doses)["next_due_date"] == "2026-05-05"


def test_a_dose_due_today_counts_as_ahead() -> None:
    """The boundary the window is measured from. Due today is due, not missed."""
    doses = [_due("Rabies", "Dose 2", "22/09/2026"), _due("Hepatitis A", "Dose 1", "26/10/2026")]

    assert _fields(doses=doses)["next_due_date"] == "2026-09-22"


def test_a_list_with_no_dates_leaves_the_flat_field_alone() -> None:
    """A history-only record states no next date. Nothing is invented and nothing the
    model did read is overwritten."""
    doses = [_given("Tdap", "Booster", "02/06/2026"), _given("Typhoid", "Single", "20/03/2026")]

    fields = _fields(next_due_date="02/10/2026", doses=doses)

    assert fields["next_due_date"] == "2026-10-02"


# --- the single-dose certificate, which must not regress ---------------------


def test_a_single_dose_certificate_is_untouched() -> None:
    """Verified on a device before this change: "Next due 02/10/2026", "Due in 10 days".
    An empty list must leave every one of those behaviours exactly as it was."""
    fields = _fields(date_given="22/09/2026", next_due_date="02/10/2026")

    assert fields["next_due_date"] == "2026-10-02"
    assert fields["doses"] == []


def test_a_certificate_stating_an_interval_still_has_it_computed() -> None:
    """The one rule this change had to be threaded around: a record saying "after 4 weeks"
    rather than a date is still turned into one in Python, not by the model."""
    fields = _fields(date_given="04/03/2026", next_due_interval="due after 4 weeks")

    assert fields["next_due_date"] == "2026-04-01"
    assert fields["next_due_interval"] == "due after 4 weeks"


def test_a_scheduled_date_beats_interval_arithmetic() -> None:
    """A date the document actually printed in its schedule is a fact; an interval added
    to a date is a derivation. The fact wins, as a printed flat date already did."""
    fields = _fields(
        date_given="04/03/2026",
        next_due_interval="after 4 weeks",
        doses=[_due("Rabies", "Dose 2", "27/09/2026")],
    )

    assert fields["next_due_date"] == "2026-09-27"


# --- saying so --------------------------------------------------------------


def test_a_schedule_says_how_many_doses_and_which_one_is_next() -> None:
    """The flat fields render as ONE record, and the chosen date belongs to a different
    vaccine. Unattributed, the card asserts "Hepatitis B ... next due 27/09/2026", which
    is false."""
    payload = _payload(date_given="14/08/2026", doses=SPECIMEN)
    flag = next(f for f in payload["flags"] if f["code"] == "multiple_doses")

    assert "9 doses" in flag["detail"]
    assert "Rabies (pre-exposure) Dose 2 of 3" in flag["detail"]
    assert "27/09/2026" in flag["detail"]


def test_a_history_with_no_next_date_says_that_instead() -> None:
    doses = [_given("Tdap", "Booster", "02/06/2026"), _given("Typhoid", "Single", "20/03/2026")]
    payload = _payload(doses=doses)
    flag = next(f for f in payload["flags"] if f["code"] == "multiple_doses")

    assert "2 doses" in flag["detail"]
    assert "None of them gives a date" in flag["detail"]


@pytest.mark.parametrize("doses", [[], [_due("Rabies", "Dose 2", "27/09/2026")]])
def test_one_dose_or_none_says_nothing(doses: list) -> None:
    """There is no second record to mention, so the card is already the whole document."""
    assert "multiple_doses" not in _codes(_payload(doses=doses))


def test_a_full_document_never_reads_as_empty() -> None:
    """``nothing_extracted`` means the model read nothing off the page. A document whose
    only content is its dose table has been read completely."""
    assert "nothing_extracted" not in _codes(
        build_payload(
            VACCINATIONS,
            VaccinationFields(doses=SPECIMEN),  # type: ignore[arg-type]
            today=TODAY,
        )
    )


# --- a known edge, pinned rather than claimed to be solved -------------------


def test_a_missed_dose_earlier_than_the_dose_given_withholds_the_reminder() -> None:
    """The flat pair is checked for order, and on a schedule the two belong to different
    vaccines — so an all-past schedule whose latest date precedes the first administered
    dose trips ``dates_out_of_order`` and ``filing`` withholds ``next_due_on``.

    Conservative rather than correct: the values stay readable in ``content`` and only the
    reminder is withheld, which is the rule the rest of this pipeline follows. Fixing it
    properly means the order check knowing the pair is not one record."""
    doses = [
        _given("Influenza QIV", "Annual", "12/09/2026"),
        _due("Tdap", "Booster", "05/05/2026"),
    ]
    payload = _payload(date_given="12/09/2026", doses=doses)

    assert payload["fields"]["next_due_date"] == "2026-05-05"
    assert "dates_out_of_order" in _codes(payload)
