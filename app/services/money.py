"""Amount and currency normalisation for section extractions.

The same rule dates already follow: the model transcribes what the document prints, and
Python decides what is stored. A prompt asking for "digits only" is a request, not a
guarantee, and documents print money in every shape there is — ``300,000.00``,
``3,00,000`` (Indian grouping), ``Rs.1000/-``, ``` ` 52,123.00 ```.

That backtick is not a typo. A rupee sign in a PDF's column header frequently comes
through text extraction as a stray character — ``` ` ``` in the HDFC ERGO schedule,
``?`` elsewhere — so the symbol beside the number is often unusable, and the currency has
to be read from wherever it *is* legible (a header, ``Rs``, or the amount spelled out in
words) and stored separately as a code. Storing the amount as printed is what made the
currency vanish from the app: nothing downstream can format ``` ` 300,000.00 ```.

So amounts are stored as bare decimal strings and the currency as an ISO-4217 code, and
the display side puts them together. Strings rather than floats because this is money;
``None`` rather than a guess, exactly like an unreadable date.
"""

import re
from decimal import Decimal, InvalidOperation

#: The first number-like token in a string: digits, with grouping separators or spaces
#: allowed inside, and an optional decimal part. Anchored at a digit, so any currency
#: symbol, label, or OCR debris in front of it is skipped rather than parsed.
#:
#: Whitespace is inside the class because a space is a real grouping separator —
#: ``44 304.00`` has to survive. The cost is that the run does not stop at a gap, so two
#: whole-rupee amounts printed side by side flow into one match. ``_leading_amount``
#: below is what decides where the first number ends; this only finds the candidate run.
_AMOUNT_RE = re.compile(r"\d[\d,\s]*(?:\.\d+)?")

#: Digits and the separators between them, kept apart so both can be inspected:
#: "3,00,000 5,00,000" -> groups ['3','00','000','5','00','000'], seps [',', ',', ' ', ',', ','].
_GROUP_SPLIT_RE = re.compile(r"([,\s]+)")


def _is_western_grouping(groups: list[str]) -> bool:
    """1,234,567 — a head of at most three digits, then groups of exactly three."""
    return len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:])


def _is_indian_grouping(groups: list[str]) -> bool:
    """3,00,000 — a head of at most two, then pairs, then a final group of three."""
    return (
        len(groups) >= 2
        and len(groups[0]) <= 2
        and len(groups[-1]) == 3
        and all(len(g) == 2 for g in groups[1:-1])
    )


def _leading_amount(matched: str) -> str:
    """The FIRST number in a matched run, dropping anything that follows it.

    A policy schedule flattened to text puts adjacent cells on one line, so a sum insured
    and the column beside it arrive as ``"3,00,000 5,00,000"``. Whitespace is a legal
    grouping separator, so the regex above swallows both and ``3,00,000`` becomes
    ``300000500000`` — three lakh stored as thirty thousand crore. Every check downstream
    passes it: it is a string, it is short enough, ``Decimal`` parses it, and a sum insured
    has no natural ceiling to test against.

    Two signals are needed, because each alone gets a real case wrong.

    **A single number never mixes separator kinds.** ``3,00,000 5,00,000`` groups commas
    and then a space; ``1,234,567 890`` does the same. Group lengths cannot see this —
    the latter is perfectly valid Western grouping if you only count digits — but the
    change from comma to space is exactly where the next column starts.

    **Consistent separators are not enough either.** ``300000 12345`` uses one space and
    nothing else, so nothing changes kind; what rules it out is that ``300000`` is
    ungrouped, and no grouping grammar admits a further group after an ungrouped head.

    So: cut where the separator kind changes, then keep the longest remaining prefix that
    forms one validly grouped number.

    The leading number is the right one to keep: the columns are read left to right and
    the amount asked for is the left one. Refusing outright would lose a figure the
    document genuinely prints, which is its own kind of wrong on a cover amount.
    """
    integer_run, _, decimals = matched.partition(".")
    parts = [p for p in _GROUP_SPLIT_RE.split(integer_run.strip()) if p]
    if not parts:
        return ""
    all_groups = parts[::2]
    groups = all_groups
    separators = ["," if "," in sep else " " for sep in parts[1::2]]

    # Stop at the first separator that is not the kind this number started with.
    if separators:
        first = separators[0]
        for index, sep in enumerate(separators):
            if sep != first:
                groups = groups[: index + 1]
                break

    kept = groups[:1]
    for size in range(len(groups), 1, -1):
        head = groups[:size]
        if _is_western_grouping(head) or _is_indian_grouping(head):
            kept = head
            break

    number = "".join(kept)
    # The decimal part sits at the end of the WHOLE run, so it belongs to the last number
    # in it. If anything at all was cut — by the separator change or by the grammar — the
    # decimals are not this number's, and appending them would move its point.
    return f"{number}.{decimals}" if decimals and kept == all_groups else number


#: What a document might print for the rupee, and what OCR/text-extraction turns it into.
#: Only currencies that could plausibly appear on a health policy here — an unrecognised
#: value becomes ``None`` rather than being invented.
_CURRENCY_ALIASES: dict[str, str] = {
    "`": "INR",  # a rupee sign the text layer could not resolve
    "₹": "INR",
    "rs": "INR",
    "rs.": "INR",
    "inr": "INR",
    "rupee": "INR",
    "rupees": "INR",
    "indian rupees": "INR",
    "indian rupee": "INR",
    "$": "USD",
    "usd": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
}

_ISO_CODE = re.compile(r"^[A-Za-z]{3}$")

#: Sorted so a scan over them is deterministic, and searched in full so two different
#: codes in one string can be detected rather than raced.
_KNOWN_CODES = sorted(set(_CURRENCY_ALIASES.values()))


def normalise_amount(raw: object) -> str | None:
    """A printed amount as a bare decimal string. ``None`` when it cannot be read.

    Takes ``object`` for the same reason ``parse_date`` does: these values come from
    ``model_dump()`` of model output, so the static type says little about what is
    actually in the field.

    A percentage is deliberately refused. ``20%`` is a co-payment share, not a sum, and a
    model that puts one in an amount field is telling us it misread the document — storing
    ``20`` there would present it as twenty rupees.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float, Decimal)):
        return str(raw)
    if not isinstance(raw, str):
        return None

    if "%" in raw:
        return None

    match = _AMOUNT_RE.search(raw)
    if match is None:
        return None

    cleaned = _leading_amount(match.group())
    if not cleaned:
        return None
    try:
        Decimal(cleaned)
    except InvalidOperation:
        return None
    return cleaned


def normalise_currency(raw: object) -> str | None:
    """An ISO-4217 code for whatever currency word or symbol the model returned.

    ``None`` when unrecognised — a wrong currency on a medical policy is worse than an
    unlabelled number, and the amount stays readable without it.
    """
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None

    alias = _CURRENCY_ALIASES.get(text.lower())
    if alias is not None:
        return alias
    if _ISO_CODE.match(text):
        return text.upper()
    # A code inside a longer answer — "Indian Rupees (INR)", "INR (`)". Matched against
    # the codes we know rather than any three letters, so a word like "THE" cannot
    # become a currency.
    #
    # TWO different codes is not a match, it is an ambiguity, and it gets the same answer
    # everything ambiguous gets here: None. This used to iterate a `set` and return
    # whichever code came out first, so "settled in INR, reinsured in USD" resolved by
    # hash order — a currency chosen by something with no opinion about the document.
    found = sorted({code for code in _KNOWN_CODES if re.search(rf"\b{code}\b", text, re.I)})
    return found[0] if len(found) == 1 else None


if __name__ == "__main__":  # pragma: no cover - self-check
    # Shapes taken from real documents. HDFC ERGO prints the rupee as a backtick in its
    # column headers, and its benefit table writes limits as "Rs.1000/-".
    assert normalise_amount("` 300,000.00") == "300000.00"
    assert normalise_amount("52,123.00") == "52123.00"
    assert normalise_amount("3,00,000") == "300000"  # Indian grouping
    assert normalise_amount("Rs.1000/-") == "1000"
    assert normalise_amount("₹ 44 304.00") == "44304.00"
    assert normalise_amount("300000.00") == "300000.00"
    assert normalise_amount(300000) == "300000"
    # Refusals: no number, a percentage in a sum field, nothing at all.
    assert normalise_amount("Not applicable") is None
    assert normalise_amount("20%") is None
    assert normalise_amount("") is None
    assert normalise_amount(None) is None

    assert normalise_currency("`") == "INR"
    assert normalise_currency("₹") == "INR"
    assert normalise_currency("Rs.") == "INR"
    assert normalise_currency("rupees") == "INR"
    assert normalise_currency("inr") == "INR"
    assert normalise_currency("AED") == "AED"  # any plain ISO code passes through
    assert normalise_currency("Indian Rupees (INR)") == "INR"  # code inside a phrase
    assert normalise_currency("Indian Rupees") == "INR"  # named, not coded
    assert normalise_currency("some other money") is None  # unrecognised: refuse
    assert normalise_currency(None) is None
    # Two different codes is an ambiguity, not a match. It used to resolve by set order.
    assert normalise_currency("settled in INR, reinsured in USD") is None

    # Two amounts printed side by side in a flattened table: keep the FIRST, which is the
    # column that was asked for. Whitespace is a legal grouping separator, so the match
    # runs straight through the gap and only the grouping grammar can say where the first
    # number ends. See _leading_amount.
    assert normalise_amount("3,00,000 5,00,000") == "300000"  # was 300000500000
    assert normalise_amount("3,00,000 50,000") == "300000"  # every group a plausible 2 or 3
    assert normalise_amount("300000 12345") == "300000"  # ungrouped, then another number
    assert normalise_amount("1,234,567 890") == "1234567"
    # A decimal belongs to the LAST number in the run, so a cut drops it with the rest.
    assert normalise_amount("3,00,000 5,00,000.50") == "300000"
    # Untouched: genuine single numbers in every grouping these documents use.
    assert normalise_amount("12,34,567") == "1234567"  # Indian
    assert normalise_amount("1,234,567.89") == "1234567.89"  # Western
    assert normalise_amount("44 304.00") == "44304.00"  # space as the separator
