"""One dosage-form vocabulary, and Spring owns it.

Three repos switch on these strings — this service emits them, Spring stores them as
``dosage_form_enum``, and the React medication wizard offers them as a search filter.
They agreed by convention and nothing asserted it, so they drifted: ``Ointment`` and
``Powder`` lived here and in the frontend long after Spring dropped them, and picking
either in the app produced a 400 from ``DosageForm.valueOf`` and no results.

This test is the guard. It reads Spring's enum from their own migration rather than
restating it, so adding a value on either side fails here instead of in production.
"""

import re
from pathlib import Path

import pytest

from app.services.medicines import _FORM_SYNONYMS, DOSAGE_FORMS, normalize_form

#: Spring's baseline migration, in the user's copy of the backend repo. The enum is
#: declared once there and mirrored by ``DosageForm.java``.
_SPRING_BASELINE = Path("D:/mhn-spring-v2/src/main/resources/db/migration/V1__baseline.sql")

_ENUM_RE = re.compile(r"CREATE\s+TYPE\s+dosage_form_enum\s+AS\s+ENUM\s*\((.*?)\)", re.S | re.I)


def _spring_dosage_forms() -> set[str]:
    text = _SPRING_BASELINE.read_text(encoding="utf-8")
    match = _ENUM_RE.search(text)
    assert match is not None, f"dosage_form_enum not found in {_SPRING_BASELINE}"
    return set(re.findall(r"'([a-z_]+)'", match.group(1)))


def test_our_vocabulary_is_exactly_springs() -> None:
    """Equality, not a subset.

    A value of ours they lack is a write that fails or a filter that 400s; a value of
    theirs we lack is a form we silently report as null. Both are drift, so this
    asserts the sets are the same rather than that ours fits inside theirs.
    """
    if not _SPRING_BASELINE.exists():
        pytest.skip(f"{_SPRING_BASELINE} not present on this machine")

    assert {form.lower() for form in DOSAGE_FORMS} == _spring_dosage_forms()


def test_every_synonym_maps_into_the_vocabulary() -> None:
    """No synonym may point at a form that is not in the set.

    This is how ``Ointment`` survived: it was removed from nobody's list because the
    synonym table still named it, and nothing checked the two agreed.
    """
    unknown = {value for value in _FORM_SYNONYMS.values() if value not in DOSAGE_FORMS}
    assert unknown == set()


def test_ointment_folds_into_cream() -> None:
    """A topical still reports a topical, rather than dropping to null.

    Both are applied to the skin, so unlike the suppository case this loses nothing
    clinical — it is a display difference, and reporting nothing would be worse.
    """
    assert normalize_form("Oint. Betnovate") == "Cream"
    assert normalize_form("E/O Ciplox") == "Cream"


def test_a_sachet_reports_no_form() -> None:
    """Powder has no home in Spring's eight, so it normalises to null.

    Null is the honest answer for a form the vocabulary does not carry — the printed
    text still survives in ``form_raw``, so nothing the document said is lost.
    """
    assert normalize_form("Sachet ORS") is None
    assert normalize_form("Powder") is None
