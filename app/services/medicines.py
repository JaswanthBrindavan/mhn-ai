"""Dosing notation -> a fixed daily schedule, in Python rather than by the model.

The prescription stage transcribes a dosing instruction exactly as printed. This turns
that text into a structure something can act on::

    {"morning": float, "afternoon": float, "evening": float, "night": float,
     "with_food": bool | None, "as_needed": bool}

Deterministic on purpose, for the same reason ``normalization`` computes abnormal flags
and ``dates`` parses dates: a rule gives the same answer every time and can be tested,
where a prompt cannot. It also keeps the arithmetic away from the model - "1/2 - 0 - 1/2"
becoming 0.5 in the morning is a calculation, and models do calculations plausibly rather
than correctly.

**Unparseable stays null.** A schedule is never invented. If the notation is not one this
recognises, ``frequency_normalized`` is None and the raw text is kept untouched for a
human to read. Reporting a wrong schedule is far worse than reporting none: a caller can
see a null, but cannot see that "1-0-1" was silently read as once daily.

``evening`` always exists so four-dose schedules map one slot per dose: ``1-1-1-1`` and
``QID`` both give 1/1/1/1. Three-times-daily notations (``1-1-1``, ``TDS``) are
morning/afternoon/night with ``evening`` at 0.0 - the slot is present either way, so the
object has one fixed shape whatever the document wrote.

Notations covered, all seen on real Indian prescriptions: the dose matrix (``1-0-1``,
``1/2-0-1/2``, the 4-slot ``1-0-0-1``), Latin abbreviations (``OD``/``BD``/``TDS``/``QID``/
``HS``, with ``AC``/``PC`` for food and ``SOS``/``PRN`` for as-needed), English prose
("1 tablet in the morning"), slot-first ("Morning-1, Night-1"), dose-first ("0.75 MG
MORNING"), clock times ("8 AM and 8 PM"), and bare rates ("twice a day").
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: A dose above this is almost certainly not a dose. Its job is to keep dates out:
#: "12-03-25" is a perfect dose matrix by shape, and only the magnitude gives it away.
MAX_DOSE_PER_SLOT = 10.0

#: The only dosage forms this service reports. Fixed product vocabulary — nothing outside
#: this set is ever returned, and it is not extended without that being a deliberate
#: decision, because every consumer switches on these nine strings.
DOSAGE_FORMS: frozenset[str] = frozenset(
    {
        "Tablet",
        "Capsule",
        "Syrup",
        "Injection",
        "Drops",
        "Cream",
        "Ointment",
        "Inhaler",
        "Powder",
    }
)

#: Printed abbreviation -> which of ``DOSAGE_FORMS`` it is.
#:
#: Normalised in Python for the same reason the schedule is: a document writes "Tab.",
#: "TAB", "TABLET" and "Tablets" for one thing, and a consumer that has to know all four
#: is a consumer that will miss the fifth. ``form_raw`` keeps the printed text, so nothing
#: is lost by mapping it.
#:
#: The Indian short forms are here because they are what prescriptions actually use and
#: none is guessable: E/D is eye drops, E/O eye ointment, N/D nasal drops, and a Rotacap
#: is a dry-powder inhaler rather than a capsule to swallow.
#:
#: **Only genuine equivalents are mapped.** A vial and an ampoule really are Injection; a
#: suspension really is taken like a Syrup; a sachet really is Powder. Forms with no
#: honest home in the nine — patch, suppository, pessary, gargle, shampoo, solution,
#: spray — are deliberately absent and normalise to None. Route is clinical, and a
#: suppository reported as a Tablet would be a swallowing instruction for something that
#: must not be swallowed. Null says "not one of ours"; a nearest guess says something
#: false, and the printed text survives in ``form_raw`` either way.
_FORM_SYNONYMS: dict[str, str] = {
    "tab": "Tablet", "tabs": "Tablet", "tablet": "Tablet", "tablets": "Tablet",
    "dt": "Tablet", "divitab": "Tablet", "divitabs": "Tablet",
    "cap": "Capsule", "caps": "Capsule", "capsule": "Capsule", "capsules": "Capsule",
    # Oral liquids. A suspension is dosed and taken exactly as a syrup is.
    "syp": "Syrup", "syrup": "Syrup", "syrups": "Syrup", "elixir": "Syrup",
    "susp": "Syrup", "suspension": "Syrup",
    # Anything delivered by needle or line.
    "inj": "Injection", "injection": "Injection", "injections": "Injection",
    "vial": "Injection", "ampoule": "Injection", "amp": "Injection",
    "iv": "Injection", "im": "Injection", "sc": "Injection",
    "infusion": "Injection", "drip": "Injection", "pen": "Injection",
    # Eye and nasal drops are both Drops; the site is not part of the vocabulary.
    "drop": "Drops", "drops": "Drops",
    "e/d": "Drops", "ed": "Drops", "eyedrops": "Drops",
    # Bare "nasal" is deliberately absent: it qualifies a route, not a device, and
    # reading it first would turn "Nasal Spray" into Drops.
    "n/d": "Drops", "nd": "Drops",
    "cream": "Cream",
    "ointment": "Ointment", "oint": "Ointment", "e/o": "Ointment", "eo": "Ointment",
    "inhaler": "Inhaler", "rotacap": "Inhaler", "respule": "Inhaler",
    "nebuliser": "Inhaler", "nebulizer": "Inhaler", "mdi": "Inhaler",
    "powder": "Powder", "granules": "Powder", "sachet": "Powder", "sachets": "Powder",
}

#: Splits a printed form into candidate tokens. Keeps "e/d" and "n/d" whole, since the
#: slash is part of the abbreviation rather than a separator.
_FORM_TOKEN_RE = re.compile(r"[a-z]+/[a-z]+|[a-z]+")

#: Which slot an hour on the clock falls in, as [from, until) in 24-hour time.
#: Anything outside these - 21:00 to 04:59 - is night.
_CLOCK_SLOT_BOUNDS = ((5, 12), (12, 17), (17, 21))

FRACTION_VALUES = {"½": 0.5, "¼": 0.25, "¾": 0.75}

WORD_NUMBERS = {
    "half": 0.5, "one": 1.0, "once": 1.0, "a": 1.0, "an": 1.0, "two": 2.0,
    "twice": 2.0, "three": 3.0, "thrice": 3.0, "four": 4.0, "five": 5.0, "six": 6.0,
}

TIME_SLOT_INDEX = {
    "morning": 0, "am": 0, "breakfast": 0,
    "noon": 1, "afternoon": 1, "midday": 1, "lunch": 1,
    "evening": 2, "eve": 2,
    "night": 3, "bedtime": 3, "dinner": 3, "pm": 3,
}

#: Latin abbreviation -> (morning, afternoon, evening, night). ``None`` means the token
#: says nothing about the schedule itself (AC/PC/SOS only modify one).
_LATIN_SCHEDULES: dict[str, tuple[float, float, float, float] | None] = {
    "od": (1.0, 0.0, 0.0, 0.0),
    "bd": (1.0, 0.0, 0.0, 1.0),
    "bid": (1.0, 0.0, 0.0, 1.0),
    "tds": (1.0, 1.0, 0.0, 1.0),
    "tid": (1.0, 1.0, 0.0, 1.0),
    "qid": (1.0, 1.0, 1.0, 1.0),
    "qds": (1.0, 1.0, 1.0, 1.0),
    "hs": (0.0, 0.0, 0.0, 1.0),
    "qhs": (0.0, 0.0, 0.0, 1.0),
    "stat": (1.0, 0.0, 0.0, 0.0),
    "sos": None,
    "prn": None,
    "ac": None,
    "pc": None,
}

_AS_NEEDED = {"sos", "prn"}
_BEFORE_FOOD = {"ac"}
_AFTER_FOOD = {"pc"}

_UNIT = (
    r"(?:(?:mcg|[µμ]g|mg|ml|gm|g|iu|meq|units?)(?![A-Za-z])"
    r"|%\s*(?:w\s*/\s*[wv]|v\s*/\s*v)?)"
)
_DOSE_COUNT = r"(?:\d+(?:\.\d+)?|\d\s*/\s*\d|[½¼¾]|half|one|two|three|four|five)"
_COUNTABLE_UNIT = (
    r"(?:tablets?|tabs?|capsules?|caps?|puffs?|drops?|spoons?(?:ful)?|teaspoons?|"
    r"tsp|sachets?|units?|doses?|applications?|scoops?|pills?|inhalations?|"
    r"sprays?|pumps?)"
)
#: Wider than ``_COUNTABLE_UNIT`` because "in the <slot>" anchors it: a volume is
#: otherwise indistinguishable from a strength ("Syp Ascoril LS 10ml TDS") and must not
#: be read as a number of doses.
_DOSE_UNIT = (
    r"(?:tablets?|tabs?|capsules?|caps?|puffs?|drops?|spoons?(?:ful)?|teaspoons?|"
    r"tsp|ml|sachets?|units?|doses?|applications?|scoops?|pills?|inhalations?|"
    r"sprays?|pumps?)"
)
_SLOT_NAMES = (
    r"morning|noon|afternoon|midday|evening|night|bed\s*time|lunch|dinner|breakfast"
)
#: The separator repeats because some documents rule a line between slots rather than
#: printing one hyphen: "1----0---1" is the same instruction as "1-0-1".
#:
#: The en and em dashes in the class are deliberate and must stay: word processors
#: autocorrect a typed hyphen into one, so a good many prescriptions carry the dose matrix
#: separated by those rather than by an ASCII hyphen. Folding them to a plain hyphen here
#: would stop every one of those parsing at all.
_SLOT_SEPARATOR = r"\s*[-–—]+\s*"  # noqa: RUF001
_FRACTION = r"(?:[½¼¾]|\d\s*/\s*\d|\d{1,2}(?:\.\d+)?)"

#: The two Latin groups are kept apart because AC/PC are two letters long and collide
#: with brand names: "PERSOL AC 2.5 GEL" is a benzoyl peroxide gel, not a dose before
#: food, and reading that "AC" as dosing truncates the name to "PERSOL".
_LATIN_SCHEDULE = (
    r"o\.?d|b\.?d|b\.?i\.?d|t\.?d\.?s|t\.?i\.?d|q\.?i\.?d|q\.?d\.?s|q\.?h\.?s|"
    r"h\.?s|s\.?o\.?s|p\.?r\.?n|stat|q\.?[1-9]\d?h"
)
_LATIN_FOOD = r"a\.?c|p\.?c"

MATRIX_RE = re.compile(
    rf"(?<![\w-])({_FRACTION}){_SLOT_SEPARATOR}({_FRACTION})"
    rf"{_SLOT_SEPARATOR}({_FRACTION})"
    rf"(?:{_SLOT_SEPARATOR}({_FRACTION}))?(?![\w-])"
)
SLOT_NAME_RE = re.compile(_SLOT_NAMES, re.IGNORECASE)
LATIN_RE = re.compile(
    rf"(?<![A-Za-z])({_LATIN_SCHEDULE}|{_LATIN_FOOD})\.?(?![A-Za-z])", re.IGNORECASE
)
FOOD_RE = re.compile(
    r"(?:after|post)\s+(?:food|meals?|lunch|dinner)"
    r"|(?:before|prior\s+to)\s+(?:food|meals?)"
    r"|empty\s+stomach|with\s+(?:food|meals?)|at\s+bed\s*time|bedtime",
    re.IGNORECASE,
)
AS_NEEDED_RE = re.compile(
    r"\b(?:as|when|if)\s+(?:and\s+when\s+)?"
    r"(?:needed|need|required|require|necessary)\b"
    r"|\bwhen\s+in\s+need\b|\bon\s+demand\b",
    re.IGNORECASE,
)
#: The lookbehind keeps a printed timestamp out: in "07:25 PM" the minutes are not an
#: hour, and a document header is not a dosing instruction.
CLOCK_TIME_RE = re.compile(
    r"(?<![:.\d])(1[0-2]|0?[1-9])\s*([ap])\.?\s?m\b\.?", re.IGNORECASE
)
MEASUREMENT_UNIT_RE = re.compile(rf"^\s*{_UNIT}\s*$", re.IGNORECASE)
ENGLISH_TIMES_RE = re.compile(
    r"\b(once|twice|thrice|one|two|three|four|\d+)\s*(?:times?)?\s*"
    r"(?:daily\b|(?:in\s+)?(?:a|per|every)\s+day\b)",
    re.IGNORECASE,
)
ENGLISH_DOSE_RE = re.compile(
    rf"({_DOSE_COUNT})\s*(?:{_DOSE_UNIT})?\s*\(?s?\)?\s*"
    r"(?:in|at|during|with)\s+the\s+"
    r"(morning|noon|afternoon|midday|evening|night|bed\s*time|lunch|dinner|breakfast)",
    re.IGNORECASE,
)
ENGLISH_DOSE_COUNT_RE = re.compile(
    rf"(?<![A-Za-z0-9])({_DOSE_COUNT})\s*{_COUNTABLE_UNIT}\s*\(?s?\)?", re.IGNORECASE
)
ENGLISH_DOSE_SLOT_RE = re.compile(
    rf"(?<![A-Za-z0-9])({_DOSE_COUNT})\s*({_DOSE_UNIT}|{_UNIT})?\s*"
    rf"(?:in\s+the\s+)?({_SLOT_NAMES})\b",
    re.IGNORECASE,
)
#: En and em dashes again deliberate, for the reason given at ``_SLOT_SEPARATOR``: the
#: autocorrected form of "Morning-1" has to parse the same way as the typed one.
ENGLISH_SLOT_DOSE_RE = re.compile(
    rf"\b({_SLOT_NAMES})\s*[-–—:]\s*({_DOSE_COUNT})(?![\d/])",  # noqa: RUF001
    re.IGNORECASE,
)
ENGLISH_SLOT_LIST_RE = re.compile(
    rf"(?:\b(?:in|at|during)\s+the\s+|\(\s*)"
    rf"((?:{_SLOT_NAMES})(?:\s*(?:,|and|&|\+)\s*(?:{_SLOT_NAMES}))*)\s*\)?",
    re.IGNORECASE,
)


def normalize_form(text: str | None) -> str | None:
    """The dosage form *text* names, as one of ``DOSAGE_FORMS``, or None.

    None means "not one of the nine", never "probably a tablet". Route is clinical:
    reporting an injection as a tablet is worse than reporting nothing, and the printed
    text is kept in ``form_raw`` either way, so a reader loses nothing by this being null.

    Reads left to right and takes the first token it knows, because that is where the form
    sits in every way a document writes it — "Tab. Dolo 650", "DOLO-TABLET-650MG", "Inj.
    Monocef 1gm".
    """
    if not text:
        return None
    for token in _FORM_TOKEN_RE.findall(text.lower()):
        form = _FORM_SYNONYMS.get(token)
        if form:
            return form
    logger.info("dosage form %r is not one of the nine - left null", text)
    return None


def parse_dose(token: str) -> float | None:
    """A single dose -> its numeric value. ``"½"``, ``"1/2"``, ``"0.5"``, ``"half"``."""
    token = (token or "").strip()
    if token in FRACTION_VALUES:
        return FRACTION_VALUES[token]
    if token.lower() in WORD_NUMBERS:
        return WORD_NUMBERS[token.lower()]
    fraction = re.fullmatch(r"(\d)\s*/\s*(\d)", token)
    if fraction:
        denominator = float(fraction.group(2))
        return float(fraction.group(1)) / denominator if denominator else None
    try:
        return float(token)
    except ValueError:
        return None


def matrix_doses(match: re.Match[str]) -> list[float] | None:
    """Parsed slots of a ``MATRIX_RE`` match, or None if it is not a real schedule."""
    doses = []
    for group in match.groups():
        if group is None:
            continue
        dose = parse_dose(group)
        if dose is None or dose > MAX_DOSE_PER_SLOT:
            # Out of range: almost always a date ("12-03-25"), not a dosage.
            return None
        doses.append(dose)
    return doses or None


def find_dose_matrix(text: str) -> re.Match[str] | None:
    """First matrix in *text* that is plausibly a dosage rather than a date."""
    for match in MATRIX_RE.finditer(text or ""):
        if matrix_doses(match) is not None:
            return match
    return None


def _normalise_latin_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def normalize_frequency(frequency_raw: str | None) -> dict[str, Any] | None:
    """Convert *frequency_raw* into the normalized schedule, or None if it will not.

    The notations are tried in order of how specific they are, and the first that yields
    a schedule wins. A dose matrix is unambiguous, so it is tried first; a bare rate
    ("twice a day") says nothing about which slots, so it is tried last.
    """
    if not frequency_raw:
        return None

    text = frequency_raw.strip()
    schedule: tuple[float, float, float, float] | None = None
    with_food: bool | None = None
    as_needed = False
    saw_any_token = False

    matrix = find_dose_matrix(text)
    if matrix:
        # find_dose_matrix has already rejected out-of-range slots (dates), so this
        # cannot come back None.
        doses = matrix_doses(matrix) or []
        if len(doses) == 3:
            # morning-afternoon-night, the common form: nothing in the evening slot.
            schedule = (doses[0], doses[1], 0.0, doses[2])
            saw_any_token = True
        elif len(doses) == 4:
            schedule = (doses[0], doses[1], doses[2], doses[3])
            saw_any_token = True

    if schedule is None:
        # Time of day first, then its dose: "Morning-1, Night-1".
        slots = [0.0, 0.0, 0.0, 0.0]
        stated = False
        for match in ENGLISH_SLOT_DOSE_RE.finditer(text):
            index = TIME_SLOT_INDEX.get(re.sub(r"\s+", "", match.group(1).lower()))
            count = parse_dose(match.group(2))
            if index is None or count is None:
                continue
            slots[index] += count
            stated = True
        if stated:
            schedule = (slots[0], slots[1], slots[2], slots[3])
            saw_any_token = True

    if schedule is None:
        # The dose stated before its time of day: "0.75 MG MORNING 0.75 MG EVENING".
        slots = [0.0, 0.0, 0.0, 0.0]
        stated = False
        for match in ENGLISH_DOSE_SLOT_RE.finditer(text):
            index = TIME_SLOT_INDEX.get(re.sub(r"\s+", "", match.group(3).lower()))
            if index is None:
                continue
            unit = match.group(2) or ""
            # A measurement unit means "0.75 MG" is restating the strength: it says take
            # one dose then, not three quarters of one. The figure stays in frequency_raw.
            measured = MEASUREMENT_UNIT_RE.match(unit)
            count = 1.0 if measured else parse_dose(match.group(1))
            if count is None:
                continue
            slots[index] += count
            stated = True
        if stated:
            schedule = (slots[0], slots[1], slots[2], slots[3])
            saw_any_token = True

    if schedule is None:
        # English prose: "1 tablet(s) in the noon; 1 tablet(s) in the night".
        slots = [0.0, 0.0, 0.0, 0.0]
        stated = False
        for match in ENGLISH_DOSE_RE.finditer(text):
            count = parse_dose(match.group(1))
            index = TIME_SLOT_INDEX.get(re.sub(r"\s+", "", match.group(2).lower()))
            if count is None or index is None:
                continue
            slots[index] += count
            stated = True
        if stated:
            schedule = (slots[0], slots[1], slots[2], slots[3])
            saw_any_token = True

    if schedule is None:
        # The dose stated once, the times of day listed at the end: "Take 1 tablet(s)
        # thrice a day for 5 days after food in the morning, noon and night."
        slot_list = ENGLISH_SLOT_LIST_RE.search(text)
        if slot_list:
            count_match = ENGLISH_DOSE_COUNT_RE.search(text)
            count = parse_dose(count_match.group(1)) if count_match else 1.0
            slots = [0.0, 0.0, 0.0, 0.0]
            stated = False
            for name in SLOT_NAME_RE.findall(slot_list.group(1)):
                index = TIME_SLOT_INDEX.get(re.sub(r"\s+", "", name.lower()))
                if index is None:
                    continue
                slots[index] += count if count is not None else 1.0
                stated = True
            if stated:
                schedule = (slots[0], slots[1], slots[2], slots[3])
                saw_any_token = True

    if schedule is None:
        # Times written on the clock: "8 AM AND 8 PM", "AT 6 AM, 2PM, 10 PM". One dose
        # at each hour named; the hour picks the slot it falls in.
        slots = [0.0, 0.0, 0.0, 0.0]
        stated = False
        for match in CLOCK_TIME_RE.finditer(text):
            hour = int(match.group(1)) % 12 + (12 if match.group(2).lower() == "p" else 0)
            index = next(
                (i for i, (low, high) in enumerate(_CLOCK_SLOT_BOUNDS) if low <= hour < high),
                3,  # late night wraps past midnight
            )
            slots[index] += 1.0
            stated = True
        if stated:
            schedule = (slots[0], slots[1], slots[2], slots[3])
            saw_any_token = True

    if schedule is None:
        # A rate with no time of day: "twice a day". Spread the same way as the
        # equivalent Latin abbreviation, so both notations agree.
        times = ENGLISH_TIMES_RE.search(text)
        if times:
            count = parse_dose(times.group(1))
            equivalent = {1: "od", 2: "bd", 3: "tds", 4: "qid"}.get(
                int(count) if count is not None and count.is_integer() else 0
            )
            if equivalent:
                schedule = _LATIN_SCHEDULES[equivalent]
                saw_any_token = True

    for match in LATIN_RE.finditer(text):
        token = _normalise_latin_token(match.group(1))
        if token not in _LATIN_SCHEDULES:
            continue
        saw_any_token = True
        if token in _AS_NEEDED:
            as_needed = True
        if token in _BEFORE_FOOD:
            with_food = False
        elif token in _AFTER_FOOD:
            with_food = True
        token_schedule = _LATIN_SCHEDULES[token]
        if token_schedule and schedule is None:
            schedule = token_schedule

    if AS_NEEDED_RE.search(text):
        # The prose form of SOS/PRN: "take when in need", "as required".
        as_needed = True
        saw_any_token = True

    for match in FOOD_RE.finditer(text):
        phrase = match.group(0).lower()
        if "before" in phrase or "prior" in phrase or "empty" in phrase:
            with_food = False
            saw_any_token = True
        elif "after" in phrase or "post" in phrase or "with" in phrase:
            with_food = True
            saw_any_token = True
        elif "bed" in phrase:
            saw_any_token = True
            if schedule is None:
                schedule = (0.0, 0.0, 0.0, 1.0)

    if schedule is None:
        if as_needed:
            # SOS/PRN with no baseline schedule is still meaningful: no fixed doses.
            schedule = (0.0, 0.0, 0.0, 0.0)
        else:
            if saw_any_token:
                logger.info(
                    "frequency %r carried only a modifier, no schedule - left null",
                    frequency_raw,
                )
            else:
                logger.info("could not normalize frequency %r - left null", frequency_raw)
            return None

    return {
        "morning": float(schedule[0]),
        "afternoon": float(schedule[1]),
        "evening": float(schedule[2]),
        "night": float(schedule[3]),
        "with_food": with_food,
        "as_needed": as_needed,
    }
