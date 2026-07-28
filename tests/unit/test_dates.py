"""Date normalisation for section extractions.

Documents print dates in whatever form the issuing system chose, so these cases are
drawn from real ones: ordinals, a DICOM StudyDate, and a radiology study timestamp.
"""

from datetime import date, datetime

import pytest

from app.services.dates import display_date, in_order, iso_date, parse_date


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("31/08/2021", date(2021, 8, 31)),  # day-first
        ("08/31/2021", date(2021, 8, 31)),  # month-first: 31 is an impossible month
        ("31-08-2021", date(2021, 8, 31)),
        ("2021-08-31", date(2021, 8, 31)),
        ("31.08.2021", date(2021, 8, 31)),
        ("31 August 2021", date(2021, 8, 31)),
        ("31 Aug 2021", date(2021, 8, 31)),
        ("August 31, 2021", date(2021, 8, 31)),
        ("Aug 31 2021", date(2021, 8, 31)),
    ],
)
def test_parses_the_common_printed_forms(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("31st August 2021", date(2021, 8, 31)),
        ("1st Jan 2026", date(2026, 1, 1)),
        ("August 31st, 2021", date(2021, 8, 31)),
        ("  31st  August  2021  ", date(2021, 8, 31)),
        ("31 August, 2021", date(2021, 8, 31)),
    ],
)
def test_parses_ordinals_commas_and_padding(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("31/08/2021 18:11:28", date(2021, 8, 31)),  # radiology study timestamp
        ("2021-08-31T18:11:28", date(2021, 8, 31)),
        ("2021-08-31T18:11:28Z", date(2021, 8, 31)),
        ("2021-08-31T18:11:28+05:30", date(2021, 8, 31)),
        ("31-08-2021 06:11 PM", date(2021, 8, 31)),
        ("31/08/2021, 18:11", date(2021, 8, 31)),
    ],
)
def test_drops_a_trailing_time(raw, expected):
    """A timestamp must not make the whole value unparseable — only the date is stored."""
    assert parse_date(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("20210831", date(2021, 8, 31)),  # DICOM StudyDate
        ("31/08/21", date(2021, 8, 31)),
        ("31-08-21", date(2021, 8, 31)),
        ("31 Aug 21", date(2021, 8, 31)),
    ],
)
def test_parses_compact_and_two_digit_years(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "31/08/2021",
        "08/31/2021",
        "2021-08-31",
        "31-08-2021",
        "31 Aug 2021",
        "31/08/2021 18:11:28",
    ],
)
def test_four_digit_years_are_not_caught_by_two_digit_patterns(raw):
    assert parse_date(raw) == date(2021, 8, 31)


@pytest.mark.parametrize(
    "raw",
    [
        "garbage",
        "",
        None,
        "2026",
        "18:11:28",
        "12345678",
        "20211301",
        "Q3 2021",
        "the 5th of never",
    ],
)
def test_unreadable_values_are_none_not_a_guess(raw):
    assert parse_date(raw) is None


@pytest.mark.parametrize("raw", [0, False, [], {}, 12345, 3.5])
def test_non_string_input_is_not_a_date(raw):
    assert parse_date(raw) is None


def test_datetime_returns_a_pure_date():
    """datetime subclasses date; returning one would break later date comparisons."""
    parsed = parse_date(datetime(2021, 8, 31, 18, 11, 28))
    assert type(parsed) is date
    assert parsed < date(2022, 1, 1)


def test_iso_date_normalises_every_form_to_one():
    for raw in ("03/07/2014", "3rd July 2014", "2014-07-03", "20140703"):
        assert iso_date(raw) == "2014-07-03"
    assert iso_date("garbage") is None
    assert iso_date(None) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2021-08-31", "31/08/2021"),
        ("31 August 2021", "31/08/2021"),
        ("28th July 2026", "28/07/2026"),
        ("31/08/2021 18:11:28", "31/08/2021"),
        (date(2021, 8, 31), "31/08/2021"),
        ("", None),
        (None, None),
    ],
)
def test_display_date_is_numeric_and_day_first(raw, expected):
    assert display_date(raw) == expected


def test_display_date_keeps_something_it_cannot_read():
    """Blanking it would hide the problem; leaving it lets a human fix it."""
    assert display_date("garbage") == "garbage"


@pytest.mark.parametrize(
    ("earlier", "later"),
    [
        ("01/01/2026", "01/01/2027"),
        ("27/07/2026", "27/07/2026"),  # same day is a valid one-day period
        ("01/01/2026", ""),  # nothing to compare
        ("", "01/01/2027"),
        ("garbage", "01/01/2027"),  # unreadable is not an ordering failure
    ],
)
def test_in_order_accepts_valid_or_uncomparable_pairs(earlier, later):
    assert in_order(earlier, later) is True


def test_in_order_rejects_an_inverted_pair():
    assert in_order("29th July 2027", "27th July 2026") is False
