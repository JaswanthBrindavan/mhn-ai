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
_AMOUNT_RE = re.compile(r"\d[\d,\s]*(?:\.\d+)?")

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

    cleaned = re.sub(r"[,\s]", "", match.group())
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
    for code in set(_CURRENCY_ALIASES.values()):
        if re.search(rf"\b{code}\b", text, re.IGNORECASE):
            return code
    return None


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
