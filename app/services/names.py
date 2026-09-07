"""Compare the patient name printed on a document with an account holder's name.

Deterministic Python, for the same reason `normalization` computes abnormal flags and
`dates` parses dates: a rule gives the same answer every run and can be tested, where a
prompt cannot. The model transcribes the name; nothing here asks it what the name means.

**Every ambiguous rule fails towards MISMATCH.** A false match files a stranger's document
into someone's medical history, where nothing downstream can detect it. A false mismatch
shows the user a dialog and costs one tap. The two errors are not comparable, so the
tolerances below are deliberately tight — `RAJ` is not accepted as `RAJESH`.

UNKNOWN is a third verdict rather than a flavour of mismatch, and it carries the feature:
plenty of real documents print no name at all (a bare X-ray, most vaccination cards), and
asking about those trains people to dismiss the dialog that matters.
"""

import re
from collections.abc import Sequence
from enum import IntEnum, StrEnum


class NameVerdict(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    #: No name on the document, or nothing left after normalisation. Not a disagreement.
    UNKNOWN = "unknown"


#: Titles that precede a name on Indian medical documents. Stripped, never compared.
_HONORIFICS = frozenset(
    {"MR", "MRS", "MS", "MISS", "DR", "MASTER", "SMT", "SHRI", "SRI", "BABY", "MX"}
)

#: Words that appear where a name should be but name nobody.
_PLACEHOLDERS = frozenset({"SELF", "PATIENT", "NAME", "NA", "NIL", "UNKNOWN", "NONE"})

#: "B/O SUNITA", "S/O RAMESH", "BABY OF SUNITA" — a relation to the named person, not a
#: name. Removed before tokenising so the person actually named survives.
_RELATIONAL = re.compile(r"\b[BSDWC]\s*/\s*O\b|\b(?:BABY|SON|DAUGHTER|WIFE)\s+OF\b")

_NOT_ALNUM = re.compile(r"[^A-Z0-9]+")

#: Below this length an edit-distance allowance would match unrelated short names, so
#: OCR tolerance applies only to tokens with enough substance to be distinctive.
_MIN_FUZZY_LENGTH = 4


def normalise(raw: str | None) -> list[str]:
    """A name reduced to comparable tokens. Empty when nothing usable remains."""
    if not raw:
        return []
    text = _RELATIONAL.sub(" ", raw.upper())
    text = _NOT_ALNUM.sub(" ", text)
    return [
        token for token in text.split() if token not in _HONORIFICS and token not in _PLACEHOLDERS
    ]


def _within_one_edit(a: str, b: str) -> bool:
    """True when one substitution, insertion or deletion turns a into b."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) <= 1
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    i = j = edits = 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] != longer[j]:
            edits += 1
            if edits > 1:
                return False
            j += 1
            continue
        i += 1
        j += 1
    return True


class _Agreement(IntEnum):
    """How two tokens agreed, strongest first — the order `min` sorts by when pairing.

    The kind matters, not just the fact: an initial corroborates almost nothing, a fuzzy
    match corroborates less than the letters suggest, and only an exact match is evidence
    on its own. `compare` counts the kinds rather than making extra passes.
    """

    EXACT = 0
    FUZZY = 1
    INITIAL = 2
    NONE = 3


def _tokens_agree(a: str, b: str) -> _Agreement:
    if a == b:
        return _Agreement.EXACT
    # An initial stands for a full token: "R" matches "RAJESH".
    if len(a) == 1 or len(b) == 1:
        return _Agreement.INITIAL if a[0] == b[0] else _Agreement.NONE
    if len(a) >= _MIN_FUZZY_LENGTH and len(b) >= _MIN_FUZZY_LENGTH and _within_one_edit(a, b):
        return _Agreement.FUZZY
    return _Agreement.NONE


def compare(document_name: str | None, account_name: str | None) -> NameVerdict:
    """Does the name printed on a document belong to this account holder?

    UNKNOWN when either side yields no usable tokens — including an account with no
    readable name, which is our gap and must never be reported as the user's mismatch.

    A match requires every token of the SHORTER name to find a partner in the longer one.
    That direction is what lets "Rajesh Sharma" match "RAJESH KUMAR SHARMA" — a document
    printing a middle name the account omits — without letting a single shared token
    carry two otherwise different names.
    """
    doc = normalise(document_name)
    account = normalise(account_name)
    if not doc or not account:
        return NameVerdict.UNKNOWN

    shorter, longer = (doc, account) if len(doc) <= len(account) else (account, doc)
    remaining = list(longer)
    exact = fuzzy = 0
    for token in shorter:
        # Take the STRONGEST partner available, not merely the first: with a fuzzy budget
        # to spend, pairing SHARMA with a SHARDA sitting earlier in the list would burn it
        # for nothing.
        best = min(((_tokens_agree(token, o), o) for o in remaining), default=None)
        if best is None or best[0] is _Agreement.NONE:
            return NameVerdict.MISMATCH
        kind, partner = best
        exact += kind is _Agreement.EXACT
        fuzzy += kind is _Agreement.FUZZY
        remaining.remove(partner)

    # Shared initials are not evidence of identity. "R S" pairs against Rajesh Sharma,
    # Ramesh Singh, Rohit Shukla and Rani Sengupta alike, and initials-only name fields
    # are common on Indian forms — so a match needs at least one token that agreed on
    # more than a first letter, and specifically one that agreed EXACTLY. Fuzzy agreement
    # alone is not corroboration either: "R SHARDA" against "Rajesh Sharma" is a
    # different family, not a misread. One damaged token is what OCR does; two is drift.
    if exact == 0 or fuzzy > 1:
        return NameVerdict.MISMATCH
    # Known and accepted residual: "RAJESH SHARDA" still matches "Rajesh Sharma" — RAJESH
    # is the exact match, SHARDA the one permitted fuzzy token. It is structurally
    # identical to the OCR case "RAJESH KUMAF SHARMA", so no cheap rule separates them,
    # and dropping edit distance altogether would fail every genuine misread instead.
    return NameVerdict.MATCH


def compare_all(document_name: str | None, account_names: Sequence[str | None]) -> NameVerdict:
    """The verdict across every name one account answers to — its own plus its aliases.

    MATCH if any of them matches, which is the whole point: a user who has confirmed
    "P Suresh Babu" once should not be asked again on the next document printing it.

    Otherwise the strongest thing anything said. MISMATCH beats UNKNOWN, so an account
    whose aliases include an unreadable entry still reports the real disagreement rather
    than being softened into "we could not tell" — every ambiguous rule in this module
    fails towards MISMATCH, and this one is no exception.

    An empty sequence is UNKNOWN, not MISMATCH: nothing was compared.
    """
    verdicts = [compare(document_name, name) for name in account_names]
    if NameVerdict.MATCH in verdicts:
        return NameVerdict.MATCH
    if NameVerdict.MISMATCH in verdicts:
        return NameVerdict.MISMATCH
    return NameVerdict.UNKNOWN


def matches_any(document_name: str | None, candidates: dict[str, str]) -> list[str]:
    """Which candidates the document's name matches, by id.

    The candidate list is supplied by Spring, already filtered to people the caller may
    write to. This function makes no access decision and reads no table — it compares
    strings. See the spec's D8.
    """
    if not normalise(document_name):
        return []
    return [
        key for key, name in candidates.items() if compare(document_name, name) is NameVerdict.MATCH
    ]
