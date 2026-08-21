"""The matcher's whole contract is this table.

Every rule fails towards MISMATCH: a false match is invisible to everyone, a false
mismatch costs the user one tap.
"""

import pytest

from app.services.names import NameVerdict, _within_one_edit, compare, matches_any, normalise

MATCHES = [
    ("MR. RAJESH KUMAR SHARMA", "Rajesh Sharma", "honorific + extra middle token"),
    ("SHARMA RAJESH", "Rajesh Sharma", "order-free"),
    ("R K SHARMA", "Rajesh Kumar Sharma", "initials"),
    ("RAJESH KUMAF SHARMA", "Rajesh Kumar Sharma", "OCR, edit distance 1"),
    ("BABY OF SUNITA DEVI", "Sunita Devi", "relational prefix"),
    ("B/O SUNITA DEVI", "Sunita Devi", "abbreviated relational prefix"),
    ("SMT. SUNITA DEVI", "Sunita Devi", "Indian honorific"),
    ("rajesh  kumar   sharma", "Rajesh Kumar Sharma", "whitespace and case"),
    ("SUNITA DEVI", "Sunita Devi", "exact"),
    ("RAJESH SHARDA", "Rajesh Sharma", "accepted residual: one exact + the one fuzzy token"),
]

MISMATCHES = [
    ("PRIYA MENON", "Rajesh Sharma", "different person"),
    ("RAJ SHARMA", "Rajesh Sharma", "RAJ/RAJESH is distance 3 — not a nickname matcher"),
    ("SUNITA DEVI", "Sunita Sharma", "one token agrees, one does not"),
    ("ANIL KUMAR", "Sunita Devi", "nothing in common"),
    ("R S", "Rajesh Sharma", "initials only — no full token corroborates"),
    ("R S", "Ramesh Singh", "the same initials fit an unrelated person"),
    ("R S", "Rohit Shukla", "and another"),
    ("R S", "Rani Sengupta", "and another"),
    ("R", "Rajesh", "a lone initial matches almost anyone"),
    ("R SHARDA", "Rajesh Sharma", "an initial plus a fuzzy token is no exact agreement"),
    ("SHARDA", "Sharma", "fuzzy alone is not corroboration"),
    ("RAJESH KUMAF SHARNA", "Rajesh Kumar Sharma", "two fuzzy tokens is drift, not a misread"),
]

UNKNOWNS = [
    (None, "Rajesh Sharma", "nothing extracted"),
    ("", "Rajesh Sharma", "empty"),
    ("   ", "Rajesh Sharma", "whitespace only"),
    ("SELF", "Rajesh Sharma", "placeholder"),
    ("PATIENT NAME", "Rajesh Sharma", "placeholder tokens only"),
    ("---", "Rajesh Sharma", "punctuation only"),
    ("MR.", "Rajesh Sharma", "honorific only"),
]


@pytest.mark.parametrize(("doc", "account", "why"), MATCHES)
def test_match(doc: str, account: str, why: str) -> None:
    assert compare(doc, account) is NameVerdict.MATCH, why


@pytest.mark.parametrize(("doc", "account", "why"), MISMATCHES)
def test_mismatch(doc: str, account: str, why: str) -> None:
    assert compare(doc, account) is NameVerdict.MISMATCH, why


@pytest.mark.parametrize(("doc", "account", "why"), UNKNOWNS)
def test_unknown(doc: str, account: str, why: str) -> None:
    assert compare(doc, account) is NameVerdict.UNKNOWN, why


def test_unreadable_account_name_is_unknown_not_mismatch() -> None:
    """An account with no usable name is our gap, not the user's. Never accuse them."""
    assert compare("RAJESH SHARMA", None) is NameVerdict.UNKNOWN
    assert compare("RAJESH SHARMA", "  ") is NameVerdict.UNKNOWN


def test_single_token_names_compare_whole() -> None:
    assert compare("SUNITA", "Sunita") is NameVerdict.MATCH
    assert compare("SUNITA", "Rajesh") is NameVerdict.MISMATCH


def test_initial_only_overlap_is_not_a_match() -> None:
    """One shared initial is not evidence. R. MENON is not Rajesh Sharma."""
    assert compare("R MENON", "Rajesh Sharma") is NameVerdict.MISMATCH


def test_a_match_needs_one_full_token_that_agrees_exactly() -> None:
    """Initials and near-misses corroborate; only an exact token is evidence."""
    assert compare("R K SHARMA", "Rajesh Kumar Sharma") is NameVerdict.MATCH
    assert compare("RAJESH KUMAF SHARMA", "Rajesh Kumar Sharma") is NameVerdict.MATCH
    assert compare("R K SHARDA", "Rajesh Kumar Sharma") is NameVerdict.MISMATCH


@pytest.mark.parametrize(
    ("a", "b", "expected", "why"),
    [
        ("KUMAR", "KUMARS", True, "insertion at the end"),
        ("UMAR", "KUMAR", True, "insertion at the front"),
        ("KUAR", "KUMAR", True, "insertion in the middle"),
        ("KUMARS", "KUMAR", True, "deletion (the same pair, other way round)"),
        ("KUMAR", "KUMARSX", False, "two insertions"),
        ("KUAR", "KUMARS", False, "an insertion and a deletion"),
        ("KUMAR", "KAMAT", False, "equal length, two substitutions"),
        ("KUMAR", "KUMAT", True, "equal length, one substitution"),
        ("KUMAR", "KUMAR", True, "identical"),
    ],
)
def test_within_one_edit(a: str, b: str, expected: bool, why: str) -> None:
    """The unequal-length walk is the insertion/deletion path the table above never hits."""
    assert _within_one_edit(a, b) is expected, why


def test_normalise_strips_and_tokenises() -> None:
    assert normalise("MR. RAJESH KUMAR SHARMA") == ["RAJESH", "KUMAR", "SHARMA"]
    assert normalise("B/O Sunita Devi") == ["SUNITA", "DEVI"]
    assert normalise("SELF") == []


def test_matches_any_returns_only_matching_ids() -> None:
    candidates = {
        "sunita-id": "Sunita Devi",
        "priya-id": "Priya Menon",
        "anil-id": "Anil Kumar",
    }
    assert matches_any("SUNITA DEVI", candidates) == ["sunita-id"]
    assert matches_any("VIKRAM RAO", candidates) == []


def test_matches_any_is_empty_for_an_unknown_name() -> None:
    """Never fan an unreadable name out across the family."""
    assert matches_any(None, {"sunita-id": "Sunita Devi"}) == []
    assert matches_any("SELF", {"sunita-id": "Sunita Devi"}) == []
