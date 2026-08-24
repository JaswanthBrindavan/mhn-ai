"""Choose the one date that means "when this document happened".

A medical document prints several dates and only one of them answers the question a
health wallet asks: when was this true of me? A lab report prints when the sample was
collected, when the lab received it and when the result was released, and they can be
days apart. A vaccination card prints the dose date and the next-due date. A bill prints
the invoice date and the payment due date.

The classification stage transcribes every labelled date it can see; this module decides
which one to keep, in Python, exactly as ``normalization`` decides abnormal flags and
``money`` decides amounts. The model is a good reader and an unreliable chooser: asked to
pick, it has no stable rule and will answer differently on two reports from the same lab.
Asked to transcribe, it is very good, and a wrong choice here is then one line in a table
rather than a prompt change and a re-run of every document.

Every ambiguous case resolves to ``None``. A null date shows the user an empty field to
fill in; a wrong date is filed, displayed, sorted on, and never questioned again.

Pure — no I/O, no model call, no database. Self-check at the bottom:
``python -m app.services.document_date``. Note the ``-m``, unlike ``normalization.py``
and ``money.py``, which are stdlib-only and so run as a plain path; this one imports
``dates`` rather than growing a second date parser beside it.
"""

from collections.abc import Sequence
from datetime import date, timedelta

from app.services.dates import parse_date

#: Label fragments that identify the event date, best first. Matched as a substring of
#: the lower-cased label printed on the document, so "Sample Collected On" hits
#: "collected". Sections absent from this table fall through to the earliest date.
_PRIORITY: dict[str, tuple[str, ...]] = {
    # Collection is when the blood was drawn, which is when the values were true of the
    # body. Received and released are the lab's own logistics.
    "reports": ("collected", "drawn", "sample", "registered", "reported", "released", "printed"),
    # The study is the scan itself; the radiologist's read can be days later.
    "scans_imaging": ("study", "scan", "performed", "exam", "report", "dictated"),
    "prescriptions": ("prescribed", "consultation", "visit", "date"),
    "vaccinations": ("given", "administered", "vaccinated", "dose", "date"),
    "bills": ("invoice", "bill", "date"),
    # A policy is found by when its cover started, not when the paperwork was issued.
    "insurance": ("period from", "policy start", "commencement", "issue", "date"),
}

#: Label fragments that disqualify a date outright, checked BEFORE the priority list.
#: The order matters: "next due date" contains "date", which is in the vaccinations
#: priority list, and a card dated by its own reminder is dated in the future on a dose
#: already given.
_NEVER: dict[str, tuple[str, ...]] = {
    "prescriptions": ("valid", "dispensed", "expiry"),
    "vaccinations": ("next due", "next dose", "due", "expiry", "batch"),
    "bills": ("due", "paid", "payment"),
    "insurance": ("expiry", "renewal", "to date", "period to"),
}

#: One day of slack. A document issued today in a timezone ahead of ours is not an error.
_FUTURE_SLACK = timedelta(days=1)


def pick(
    section: str,
    dates: Sequence[tuple[str, str]],
    *,
    today: date | None = None,
) -> tuple[date | None, str | None]:
    """The document's own date, and the label it was printed under.

    ``dates`` is ``(label, value)`` as transcribed, in page order; a label may be empty.
    ``section`` is a ``DocumentSection`` *value* rather than the enum, so this module
    needs no app imports beyond date parsing.

    Returns ``(None, None)`` when nothing resolves, which is a real answer: many bare
    X-rays, vaccination cards and bills print no date at all.
    """
    horizon = (today or date.today()) + _FUTURE_SLACK
    never = _NEVER.get(section, ())

    usable: list[tuple[str, date]] = []
    for raw_label, raw_value in dates:
        label = (raw_label or "").strip()
        lowered = label.lower()
        if any(fragment in lowered for fragment in never):
            continue
        parsed = parse_date(raw_value)
        if parsed is None or parsed > horizon:
            continue
        usable.append((label, parsed))

    if not usable:
        return None, None

    for fragment in _PRIORITY.get(section, ()):
        for label, parsed in usable:
            if fragment in label.lower():
                return parsed, label or None

    # Nothing matched a known label — an unlabelled date in a header, or a wording no
    # list anticipated. The earliest date on the page is the event in every shape seen so
    # far: administrative dates come after the thing they administer.
    label, parsed = min(usable, key=lambda pair: pair[1])
    return parsed, label or None


if __name__ == "__main__":
    assert pick("reports", [("Reported On", "15/03/2026"), ("Sample Collected", "12/03/2026")]) == (
        date(2026, 3, 12),
        "Sample Collected",
    )
    assert pick(
        "vaccinations", [("Next Due Date", "01/01/2027"), ("Date Given", "01/01/2026")]
    ) == (
        date(2026, 1, 1),
        "Date Given",
    )
    assert pick("vaccinations", [("Next Due Date", "01/01/2027")]) == (None, None)
    assert pick("bills", [("Payment Due", "30/04/2026"), ("Invoice Date", "01/04/2026")]) == (
        date(2026, 4, 1),
        "Invoice Date",
    )
    assert pick("scans_imaging", [("StudyDate", "20260308")]) == (date(2026, 3, 8), "StudyDate")
    assert pick("reports", [("", "20/03/2026"), ("", "12/03/2026")]) == (date(2026, 3, 12), None)
    assert pick("reports", [("Sample Collected", "gibberish")]) == (None, None)
    assert pick("vaccinations", [("Booster On", "01/01/2099")], today=date(2026, 8, 22)) == (
        None,
        None,
    )
    assert pick("scans_imaging", []) == (None, None)
