"""Choosing the one date that means "when this document happened"."""

from datetime import date

import pytest

from app.services.document_date import pick


def test_reports_prefer_collection_over_release():
    # The three dates a lab report prints, and they can be days apart. Only the first
    # answers "when was this true of me".
    chosen, label = pick(
        "reports",
        [
            ("Reported On", "15/03/2026"),
            ("Sample Collected", "12/03/2026"),
            ("Sample Received", "13/03/2026"),
        ],
    )
    assert chosen == date(2026, 3, 12)
    assert label == "Sample Collected"


def test_vaccination_next_due_never_wins_even_though_it_contains_date():
    # "Next Due Date" contains "date", which is in the vaccinations priority list. The
    # never-list has to beat the priority list, or every vaccination card is dated by its
    # own reminder -- a date in the future, on a dose already given.
    chosen, label = pick(
        "vaccinations",
        [("Next Due Date", "01/01/2027"), ("Date Given", "01/01/2026")],
    )
    assert chosen == date(2026, 1, 1)
    assert label == "Date Given"


def test_a_vaccination_card_printing_only_a_next_due_date_yields_nothing():
    # Rather than dating the dose by the reminder. Null shows the user an empty field;
    # a wrong date is filed, displayed, sorted on and never questioned.
    assert pick("vaccinations", [("Next Due Date", "01/01/2027")]) == (None, None)


def test_bills_ignore_due_and_paid():
    chosen, label = pick(
        "bills",
        [("Payment Due", "30/04/2026"), ("Invoice Date", "01/04/2026")],
    )
    assert chosen == date(2026, 4, 1)
    assert label == "Invoice Date"


def test_unlabelled_dates_fall_back_to_the_earliest():
    chosen, label = pick("reports", [("", "20/03/2026"), ("", "12/03/2026")])
    assert chosen == date(2026, 3, 12)
    assert label is None


def test_the_earliest_fallback_ignores_page_order():
    # min() on the parsed value, not the first one transcribed: a report can print its
    # release date in the header and its collection date in the table below.
    chosen, _ = pick("medical_condition", [("Recorded", "05/05/2026"), ("Seen", "01/05/2026")])
    assert chosen == date(2026, 5, 1)


def test_a_future_date_is_discarded():
    # No document records an event that has not happened. The commonest cause is a
    # "next due" that escaped the never-list under an unexpected label.
    assert pick("vaccinations", [("Booster On", "01/01/2099")], today=date(2026, 8, 22)) == (
        None,
        None,
    )


def test_today_is_not_in_the_future():
    # A document issued today, and one issued today in a timezone ahead of ours.
    chosen, _ = pick("bills", [("Invoice Date", "22/08/2026")], today=date(2026, 8, 22))
    assert chosen == date(2026, 8, 22)
    chosen, _ = pick("bills", [("Invoice Date", "23/08/2026")], today=date(2026, 8, 22))
    assert chosen == date(2026, 8, 23)


def test_unparseable_values_are_dropped_not_guessed():
    assert pick("reports", [("Sample Collected", "not a date")]) == (None, None)


def test_one_unparseable_value_does_not_lose_the_others():
    chosen, label = pick("reports", [("Sample Collected", "-"), ("Reported On", "15/03/2026")])
    assert chosen == date(2026, 3, 15)
    assert label == "Reported On"


def test_no_dates_at_all():
    assert pick("scans_imaging", []) == (None, None)


def test_dates_are_read_day_first():
    # Indian documents, and dates.parse_date is day-first for that reason. 03/07 is
    # 3 July, and a month-first reading would be a silently plausible wrong answer.
    chosen, _ = pick("bills", [("Invoice Date", "03/07/2026")])
    assert chosen == date(2026, 7, 3)


def test_a_dicom_study_date_is_read():
    chosen, label = pick("scans_imaging", [("StudyDate", "20260308")])
    assert chosen == date(2026, 3, 8)
    assert label == "StudyDate"


@pytest.mark.parametrize(
    ("section", "labelled", "expected", "expected_label"),
    [
        (
            "scans_imaging",
            [("Report Date", "10/05/2026"), ("Study Date", "08/05/2026")],
            date(2026, 5, 8),
            "Study Date",
        ),
        (
            "prescriptions",
            [("Valid Until", "01/06/2026"), ("Prescribed On", "01/05/2026")],
            date(2026, 5, 1),
            "Prescribed On",
        ),
        (
            "insurance",
            [("Expiry Date", "31/03/2027"), ("Period From", "01/04/2026")],
            date(2026, 4, 1),
            "Period From",
        ),
        (
            "bills",
            [("Bill Date", "02/04/2026"), ("Due Date", "30/04/2026")],
            date(2026, 4, 2),
            "Bill Date",
        ),
    ],
)
def test_each_section_prefers_its_own_event_date(section, labelled, expected, expected_label):
    chosen, label = pick(section, labelled)
    assert chosen == expected
    assert label == expected_label


def test_matching_is_case_insensitive():
    chosen, label = pick("reports", [("SAMPLE COLLECTED ON", "12/03/2026")])
    assert chosen == date(2026, 3, 12)
    assert label == "SAMPLE COLLECTED ON"


def test_priority_order_decides_not_page_order():
    # "Registered" comes before "Reported" in the list, so it wins even though the
    # reported date was transcribed first.
    chosen, label = pick(
        "reports",
        [("Reported On", "15/03/2026"), ("Registered On", "13/03/2026")],
    )
    assert label == "Registered On"
    assert chosen == date(2026, 3, 13)


def test_an_unknown_section_has_no_priority_list_and_no_never_list():
    chosen, label = pick("unknown", [("Anything", "01/05/2026")])
    assert chosen == date(2026, 5, 1)
    assert label == "Anything"
