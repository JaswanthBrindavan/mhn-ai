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
from enum import StrEnum


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


def _tokens_agree(a: str, b: str) -> bool:
    if a == b:
        return True
    # An initial stands for a full token: "R" matches "RAJESH".
    if len(a) == 1 or len(b) == 1:
        return a[0] == b[0]
    if len(a) >= _MIN_FUZZY_LENGTH and len(b) >= _MIN_FUZZY_LENGTH:
        return _within_one_edit(a, b)
    return False


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
    for token in shorter:
        partner = next((o for o in remaining if _tokens_agree(token, o)), None)
        if partner is None:
            return NameVerdict.MISMATCH
        remaining.remove(partner)

    # One shared initial is not evidence of identity: "R MENON" against "Rajesh Sharma"
    # pairs R->Rajesh and then fails on MENON, but a single-token name of one letter
    # would otherwise pass on nothing at all.
    if len(shorter) == 1 and len(shorter[0]) == 1:
        return NameVerdict.MISMATCH
    return NameVerdict.MATCH


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
