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
from calendar import monthrange
from datetime import date, datetime, timedelta

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
    # Hyphenated month abbreviation — "18-Mar-2026". Extremely common on Indian lab
    # reports and absent here until 2026-09-02, which cost more than a parse: the
    # document's own date is chosen from these (``document_date.pick``), so every
    # report from a lab printing this shape filed with a NULL date and showed the
    # reader an empty field. It cannot collide with "%d-%m-%Y" — that one needs a
    # numeric month, and this one an alphabetic name.
    "%d-%b-%Y",
    "%d-%B-%Y",
    # The same month name with SLASHES — "01/Jul/2026". Found on a production
    # report on 2026-09-03, and it cost exactly what the hyphenated shape above
    # cost: `pick` chooses the document's own date from these, so every report
    # printing this filed with a NULL date and showed the reader an empty field.
    # It cannot collide with "%d/%m/%Y" for the same reason that one cannot
    # collide with "%d-%m-%Y" — a numeric month against an alphabetic name.
    "%d/%b/%Y",
    "%d/%B/%Y",
    "%B %d %Y",
    "%b %d %Y",
    "%Y%m%d",  # DICOM StudyDate, common on imaging reports
    # Two-digit years last. Python maps 00-68 -> 2000s, 69-99 -> 1900s.
    "%d/%m/%y",
    "%d-%m-%y",
    "%d.%m.%y",
    "%d %b %y",
    # "18-Mar-26" — the same shape as above with the year abbreviated, which is how the
    # comparison table on a MedPlus cumulative report dates its columns.
    "%d-%b-%y",
    "%d/%b/%y",
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


# --- stated intervals -------------------------------------------------------
#
# A vaccination card often states WHEN the next dose is due as an interval rather than a
# date: "next dose due after 4 weeks". Asked to return the start of that window, a model
# computes date_given + 28 days and hands back a date that appears nowhere in the
# document — and for vaccinations that value is written to `vaccinations.next_due_on`,
# which drives Spring's reminder index. So a model's date arithmetic would decide when a
# parent is reminded to bring a child back.
#
# Measured: a card reading "Date given: 04/03/2026 / Next dose due after 4 weeks" came
# back with next_due_date 01/04/2026, computed. Correct that time, unverifiable in
# general, and stored indistinguishably from a printed date.
#
# So the model transcribes the phrase and this computes the date, for the same reason
# abnormal flags, dose schedules and money are computed here.

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
}  # fmt: skip

#: "4 weeks", "one month", "6-8 weeks".
_INTERVAL_RE = re.compile(
    r"(?P<count>\d+|" + "|".join(_WORD_NUMBERS) + r")\s*(?P<unit>day|week|month|year)s?\b",
    re.IGNORECASE,
)
#: A range collapses to its first number BEFORE matching: "6-8 weeks" is one interval, and
#: the regex would otherwise skip the 6 (no unit follows it) and read "8 weeks" — the last
#: day of the window rather than the first. The day a dose first becomes due is the one
#: this field means, the same rule a printed window follows.
_INTERVAL_RANGE_RE = re.compile(r"(\d+)\s*(?:-|–|—|to)\s*\d+", re.IGNORECASE)  # noqa: RUF001


def _add_months(start: date, months: int) -> date:
    """Calendar months, clamped to the end of the target month.

    31 January + 1 month is 28 February, not 3 March. Written out rather than pulled from
    dateutil: four lines against a dependency, and the clamping rule is the whole content.
    """
    total = start.month - 1 + months
    year = start.year + total // 12
    month = total % 12 + 1
    last_day = monthrange(year, month)[1]
    return date(year, month, min(start.day, last_day))


def _parse_interval(phrase: object) -> tuple[int, str] | None:
    """``(count, unit)`` from a stated interval, or None when there is not one.

    Shared by the two readers below so one phrase cannot be understood two ways: a
    duration that yields an end date and a duration that yields a day count have to agree
    about what "6-8 weeks" means, and one parse is the only way to guarantee that.
    """
    if not isinstance(phrase, str):
        return None
    match = _INTERVAL_RE.search(_INTERVAL_RANGE_RE.sub(r"\1", phrase))
    if match is None:
        return None
    raw_count = match.group("count").lower()
    # The regex admits digits or a known word and nothing else, so both branches are safe.
    count = int(raw_count) if raw_count.isdigit() else _WORD_NUMBERS[raw_count]
    return (count, match.group("unit").lower()) if count > 0 else None


#: Days per unit. A month is 30 and a year 365 -- APPROXIMATE, unlike everything else in
#: this module, and deliberately so. See ``interval_days``.
_DAYS_PER_UNIT = {"day": 1, "week": 7, "month": 30, "year": 365}

#: ``prescription_item.duration_days`` is ``int2``. A count past this is not a long course,
#: it is a misread, and letting it through would fail an INSERT rather than merely look odd.
_MAX_DURATION_DAYS = 32767


def interval_days(phrase: object) -> int | None:
    """A prescribed duration as a number of days: "5 days" -> 5, "2 weeks" -> 14.

    Feeds the End date the confirm screen prefills, which is the whole reason it exists: a
    course ends at its start plus its duration, and only that screen knows the start the
    user picked.

    **Months and years are approximated (30 and 365), which breaks this module's usual
    refuse-rather-than-guess rule on purpose.** That rule protects values nobody looks at
    -- a document date is filed, displayed, sorted on and never questioned. This one is
    prefilled into a field the user is reading and can change before saving, so an end date
    two days out is visible and correctable, while a blank one fails the commonest case
    there is: a chronic medicine written for "1 month". The raw ``duration`` text travels
    beside it either way, so nothing is lost by the approximation.
    """
    parsed = _parse_interval(phrase)
    if parsed is None:
        return None
    days = parsed[0] * _DAYS_PER_UNIT[parsed[1]]
    return days if days <= _MAX_DURATION_DAYS else None


def add_interval(start: object, phrase: object) -> str | None:
    """``start`` plus a stated interval, as an ISO date. None when either is unreadable.

    Refuses rather than guesses, like every other decision here: no start date, no
    recognisable interval, or an unknown unit all give None. A range takes its first
    number, so "6-8 weeks" is the day the dose first becomes due.
    """
    begin = parse_date(start)
    parsed = _parse_interval(phrase)
    if begin is None or parsed is None:
        return None
    count, unit = parsed

    if unit == "day":
        return (begin + timedelta(days=count)).isoformat()
    if unit == "week":
        return (begin + timedelta(weeks=count)).isoformat()
    if unit == "month":
        return _add_months(begin, count).isoformat()
    return _add_months(begin, count * 12).isoformat()
