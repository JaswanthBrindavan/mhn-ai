"""Date parsing and rendering for section extractions.

Documents print dates in whatever form the issuing system chose: day-first, month-first,
ordinals ("28th July 2026"), a DICOM ``StudyDate`` ("20210831"), or a full study
timestamp ("31/08/2021 18:11:28"). The model is asked for ``DD/MM/YYYY``, but a prompt
is a request, not a guarantee — so every date is normalised here before it is stored or
compared. Parsing failures are surfaced as ``None`` rather than a guess.

Three representations, deliberately separate:

    parse_date()   -> date        for comparisons and arithmetic
    iso_date()     -> 2021-08-31  for storage (sorts correctly, unambiguous)
    display_date() -> 31/08/2021  for anything a person reads

Day-first is tried before month-first: the documents are Indian. That makes an
ambiguous pair like ``03/07/2014`` deterministically 3 July, never 7 March.
"""

import re
from datetime import date, datetime

#: Four-digit years first, so a full year is never captured by a two-digit pattern.
#: Applied after _normalise(), so these carry no ordinals, commas, or trailing time.
_DATE_FORMATS: tuple[str, ...] = (
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y-%m-%d",
    "%d.%m.%Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%Y%m%d",  # DICOM StudyDate, common on imaging reports
    # Two-digit years last. Python maps 00-68 -> 2000s, 69-99 -> 1900s.
    "%d/%m/%y",
    "%d-%m-%y",
    "%d.%m.%y",
    "%d %b %y",
)

#: "28th July" / "1st Jan" — both printed on documents and typed by people.
_ORDINAL = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b", re.IGNORECASE)

#: A trailing clock time. Radiology reports print the study timestamp, and an ISO
#: datetime uses a T separator. Only the date is ever stored, so drop the rest rather
#: than failing to parse the whole value.
_TIME = re.compile(
    r"[T\s]+\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\s*(?:[ap]\.?m\.?)?"
    r"(?:\s*(?:Z|[+-]\d{2}:?\d{2}))?\s*$",
    re.IGNORECASE,
)

DISPLAY_FORMAT = "%d/%m/%Y"


def _normalise(value: str) -> str:
    """Strip ordinal suffixes, commas, a trailing time, and repeated whitespace."""
    text = _ORDINAL.sub(r"\1", value)
    text = text.replace(",", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return _TIME.sub("", text).strip()


def parse_date(raw: object) -> date | None:
    """Parse a date as printed on a document. ``None`` when it cannot be read.

    Takes ``object`` rather than ``str | date | None`` deliberately: the values arriving
    here come from ``model_dump()`` of model output, so the static type says little about
    what is actually in the field. Anything that is not a readable date returns ``None``.

    Never guesses: an unreadable value returns ``None`` so the caller can leave the
    field empty rather than store something invented.
    """
    if not raw:
        return None
    # datetime subclasses date, so it must be checked first — returning a datetime
    # would raise TypeError on any later `datetime < date` comparison.
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if not isinstance(raw, str):
        return None

    text = _normalise(raw)
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def iso_date(raw: object) -> str | None:
    """Render as ``YYYY-MM-DD`` for storage. ``None`` when unparseable."""
    parsed = parse_date(raw)
    return parsed.isoformat() if parsed else None


def display_date(raw: object) -> str | None:
    """Render as ``DD/MM/YYYY`` for a person to read.

    An unparseable value is returned unchanged rather than blanked, so something we
    could not read stays visible instead of silently disappearing.
    """
    if raw is None or raw == "":
        return None
    parsed = parse_date(raw)
    return parsed.strftime(DISPLAY_FORMAT) if parsed else str(raw)


def in_order(earlier: object, later: object) -> bool:
    """True unless both dates parse AND ``later`` precedes ``earlier``.

    A blank or unreadable date is not an ordering error — there is simply nothing to
    compare, which is a different problem from a genuinely inverted pair.
    """
    start = parse_date(earlier)
    end = parse_date(later)
    if start is None or end is None:
        return True
    return end >= start
