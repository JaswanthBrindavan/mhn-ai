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


# --- two amounts on one line ------------------------------------------------
#
# A policy schedule flattened to text puts adjacent cells on one line, so a sum insured
# arrives beside the column next to it. Whitespace is a legal grouping separator, so the
# amount regex runs straight through the gap: "3,00,000 5,00,000" became 300000500000 --
# three lakh stored as thirty thousand crore, past every downstream check, because a sum
# insured has no natural ceiling to test against.


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        # Both columns whole-rupee, which is how Indian policies print a cover.
        ("3,00,000 5,00,000", "300000"),
        # Every group here is a plausible 2 or 3, so group LENGTHS alone cannot cut it.
        # The separator changing from comma to space is what marks the next column.
        ("3,00,000 50,000", "300000"),
        # Validly Western-grouped if you only count digits -- 1,234,567 890 could be one
        # number. The mixed separator is again the tell.
        ("1,234,567 890", "1234567"),
        # Consistent separators, so nothing changes kind: what rules this out is that an
        # ungrouped head admits no further group.
        ("300000 12345", "300000"),
        # The decimals belong to the LAST number in the run, so a cut drops them too --
        # appending them would move the kept number's point.
        ("3,00,000 5,00,000.50", "300000"),
    ],
)
def test_only_the_leading_amount_is_kept(printed: str, expected: str) -> None:
    """The columns are read left to right and the asked-for amount is the left one.

    Keeping it rather than refusing is deliberate: a blank cover amount is its own kind
    of wrong, and the document does genuinely print this figure.
    """
    assert normalise_amount(printed) == expected


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        ("12,34,567", "1234567"),  # Indian grouping
        ("1,234,567", "1234567"),  # Western grouping
        ("1,234,567.89", "1234567.89"),
        ("44 304.00", "44304.00"),  # a space really is the separator here
        ("3,00,000", "300000"),
        ("300000.00", "300000.00"),
        ("` 300,000.00", "300000.00"),  # rupee sign the text layer could not resolve
        ("Rs.1000/-", "1000"),
    ],
)
def test_single_amounts_are_untouched(printed: str, expected: str) -> None:
    """The cut must not cost a genuine number. Every grouping these documents use."""
    assert normalise_amount(printed) == expected


def test_two_currencies_are_an_ambiguity_not_a_match() -> None:
    """It used to iterate a `set` and return whichever code came out first, so this
    resolved by hash order -- a currency picked by something with no opinion about the
    document. Ambiguity gets the same answer as everything else ambiguous here."""
    assert normalise_currency("settled in INR, reinsured in USD") is None
    # One code in a longer phrase is still a match.
    assert normalise_currency("Indian Rupees (INR)") == "INR"
