"""Amount and currency normalisation.

The cases are the shapes real policy documents print, not invented ones. The backtick
comes from HDFC ERGO's schedule, where the rupee sign in a column header reaches the text
layer as a stray character — which is why the currency is stored as a code rather than
kept as printed.
"""

import pytest

from app.services.money import normalise_amount, normalise_currency


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("` 300,000.00", "300000.00"),  # rupee sign lost by the text layer
        ("₹ 300,000.00", "300000.00"),
        ("3,00,000", "300000"),  # Indian grouping
        ("52,123.00", "52123.00"),
        ("Rs.1000/-", "1000"),
        ("Rs 5000/- per claim", "5000"),
        ("44 304.00", "44304.00"),  # OCR splitting a group separator into a space
        ("300000.00", "300000.00"),
        (300000, "300000"),
    ],
)
def test_amounts_reduce_to_a_bare_decimal(raw, expected):
    assert normalise_amount(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "20%",  # a co-payment share, not a sum: storing 20 would read as twenty rupees
        "1% of sum insured",
        "Not applicable",
        "",
        None,
        [],
    ],
)
def test_an_amount_that_cannot_be_read_is_null_rather_than_a_guess(raw):
    assert normalise_amount(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("`", "INR"),
        ("₹", "INR"),
        ("Rs", "INR"),
        ("Rs.", "INR"),
        ("inr", "INR"),
        ("Rupees", "INR"),
        ("Indian Rupees", "INR"),
        ("Indian Rupees (INR)", "INR"),  # a code inside a longer answer
        ("USD", "USD"),
        ("AED", "AED"),  # any plain three-letter code passes through
    ],
)
def test_currencies_resolve_to_an_iso_code(raw, expected):
    assert normalise_currency(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "money", "the policy currency", None, 42])
def test_an_unrecognised_currency_is_null(raw):
    """A wrong currency on a medical policy is worse than an unlabelled number."""
    assert normalise_currency(raw) is None


def test_a_word_inside_a_phrase_is_not_mistaken_for_a_code():
    """Searching *inside* a longer answer matches known codes only, so an ordinary
    three-letter word cannot become a currency.

    A bare three-letter token is a different case and is accepted as a code — that is
    what lets 'AED' or 'SGD' work without enumerating ISO-4217 here, and the field is
    asked for a code and nothing else.
    """
    assert normalise_currency("per claim basis") is None
    assert normalise_currency("the sum assured") is None
