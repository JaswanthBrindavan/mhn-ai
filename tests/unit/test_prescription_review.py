"""Three findings from the prescriptions review, each reproduced before it was fixed.

The stage has its own pipeline rather than a ``SectionSpec``, which is a deliberate
decision — a prescription is a layout and its names need a guard no other section has. The
cost is that every improvement made to the generic stage has to be applied here by hand,
and two of these three are that cost coming due.
"""

from app.integrations.ai.base import DocumentPayload
from app.services.prescriptions import (
    PrescribedMedicine,
    PrescriptionFields,
    _build_payload,
    _has_drug_identity,
    _verify_against_document,
)
from tests.support.pdfs import text_pdf


def _doc(text: str) -> DocumentPayload:
    return DocumentPayload(data=text_pdf(text), content_type="application/pdf", filename="rx.pdf")


def _med(written: str, clean: str) -> PrescribedMedicine:
    return PrescribedMedicine(name_as_written=written, name_clean=clean)


def _run(document: DocumentPayload, rows: list[PrescribedMedicine]) -> dict:
    kept, rejected, loose, verified, _ = _verify_against_document(document, rows)
    result = PrescriptionFields(medicines=rows, prescribed_date=None, prescriber=None)
    return _build_payload(result, kept, rejected, verified, loose)


# --- a two-character drug name ----------------------------------------------


def test_a_short_name_with_a_digit_names_a_drug() -> None:
    """ "D3" is two characters and is one of the most commonly prescribed things in India.

    The length test alone read "short" as "not a drug" and dropped it. No dosage form is
    written with a digit in it, so a digit is enough to tell a drug from the furniture.
    """
    assert _has_drug_identity("D3") is True
    assert _has_drug_identity("K2") is True
    # And the thing the check exists for still fails it.
    assert _has_drug_identity("Tablet") is False
    assert _has_drug_identity("Cap") is False


def test_a_weekly_vitamin_d_dose_survives_the_guard() -> None:
    """End to end on a document that prints it. Reproduced before the fix: the model
    returned D3, the page printed D3, and the payload had neither the medicine nor any
    flag mentioning it."""
    payload = _run(
        _doc("Tab D3 60K weekly\nTab. DOLO 650 1-0-1\n"),
        [_med("Tab D3 60K", "D3"), _med("Tab. DOLO 650", "DOLO")],
    )

    assert [m["name_clean"] for m in payload["fields"]["medicines"]] == ["D3", "DOLO"]
    assert payload["flags"] == []


# --- the silent drop --------------------------------------------------------


def test_a_row_that_names_no_drug_is_reported_rather_than_vanishing() -> None:
    """The one path in this stage that removed a medicine and recorded nothing.

    Everything else here is scrupulous about it — a name the page lacks is flagged, a
    partial match is flagged, unreadable dosing is flagged. This one deleted in silence,
    so a filtered prescription and a genuinely empty one looked identical to every reader.
    """
    payload = _run(
        _doc("Tab. DOLO 650 1-0-1\nTablet\n"),
        [_med("Tab. DOLO 650", "DOLO"), _med("Tablet", "Tablet")],
    )

    assert [m["name_clean"] for m in payload["fields"]["medicines"]] == ["DOLO"]
    flag = next(f for f in payload["flags"] if f["code"] == "names_not_on_document")
    assert flag["names"] == ["Tablet"]
    # Wording true of BOTH reasons that land here: a name the page never mentions may be an
    # invention, while "Tablet" is printed all over the page and simply names no drug.
    assert "could not be matched to a medicine" in flag["detail"]


def test_dropped_rows_are_still_reported_when_nothing_survives() -> None:
    """A document whose every row was filtered must not read as a document with no
    medicines on it."""
    payload = _run(_doc("Tablet\nCapsule\n"), [_med("Tablet", "Tablet"), _med("Cap", "Cap")])

    assert payload["fields"]["medicines"] == []
    assert [f["code"] for f in payload["flags"]] == ["names_not_on_document"]


# --- the date ---------------------------------------------------------------


def test_the_prescription_date_is_normalised_like_every_other_section() -> None:
    """This stage inherited neither half of what a SectionSpec gets — no date rule in the
    prompt, no iso_date() on the way out — and stored whatever the model returned.

    It is visible: the app declares this field a date and its formatter renders anything
    that is not ISO unchanged, so one prescription read "Dated: 23rd December 2021" and
    the next "Dated: 15/01/2026", decided by how the document happened to print it.
    """
    result = PrescriptionFields(
        medicines=[], prescribed_date="23rd December 2021", prescriber="Dr A"
    )

    assert _build_payload(result, [], [])["fields"]["prescribed_date"] == "2021-12-23"


def test_an_unreadable_prescription_date_becomes_null_rather_than_a_guess() -> None:
    result = PrescriptionFields(medicines=[], prescribed_date="sometime last winter")

    assert _build_payload(result, [], [])["fields"]["prescribed_date"] is None
