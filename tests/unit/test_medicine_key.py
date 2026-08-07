"""The per-medicine key: what a confirm screen records when a user ticks a medicine.

It has to survive a re-run of the same document, distinguish two lines that transcribe
identically, and change when the transcription changes. The failure it exists to prevent
is confirming one medicine on a consolidated bill and later having that reference address
a different one.
"""

from typing import Any

from app.services.prescriptions import PrescribedMedicine, PrescriptionFields, _build_payload


def medicine(**kwargs: Any) -> PrescribedMedicine:
    base: dict[str, Any] = {
        "name_as_written": "Tab. DOLO 650",
        "name_clean": "DOLO",
        "strength": "650mg",
        "frequency_raw": "1-0-1 after food",
        "duration": "5 days",
    }
    base.update(kwargs)
    return PrescribedMedicine(**base)


def keys_for(*rows: PrescribedMedicine) -> list[str]:
    result = PrescriptionFields(medicines=list(rows), prescribed_date=None, prescriber=None)
    payload = _build_payload(result, list(rows), [])
    return [m["key"] for m in payload["fields"]["medicines"]]


def test_the_same_read_gives_the_same_key() -> None:
    # SQS is at-least-once and the payload is upserted, so a redelivery re-runs this whole
    # stage. A key that changed on re-run would dangle every confirmation already recorded.
    assert keys_for(medicine()) == keys_for(medicine())


def test_the_same_drug_billed_twice_gets_two_keys() -> None:
    """The case position-in-list gets wrong.

    A consolidated bill staples several purchases together and the prompt requires one
    entry per printed line even when name and strength repeat exactly. Two identical lines
    are two purchases and must be separately confirmable.
    """
    first, second, third = keys_for(medicine(), medicine(), medicine())
    assert len({first, second, third}) == 3


def test_a_changed_transcription_changes_the_key() -> None:
    """Correct, not a flaw: if the model now reads a different dose, the thing the user
    confirmed genuinely no longer exists and the reference should not silently follow."""
    baseline = keys_for(medicine())
    assert keys_for(medicine(strength="500mg")) != baseline
    assert keys_for(medicine(frequency_raw="1-0-0")) != baseline
    assert keys_for(medicine(name_as_written="Tab. DOLO 1000")) != baseline


def test_a_different_medicine_on_the_same_page_gets_its_own_key() -> None:
    dolo, pan = keys_for(medicine(), medicine(name_as_written="Cap. PAN 40", name_clean="PAN"))
    assert dolo != pan


def test_order_of_identical_lines_is_what_separates_them() -> None:
    # Deliberately asserted: the ordinal is the only thing distinguishing byte-identical
    # lines, so the keys must follow printed order rather than being interchangeable.
    a, b = keys_for(medicine(), medicine())
    assert keys_for(medicine(), medicine()) == [a, b]


def test_medicine_id_is_always_present_and_null_for_now() -> None:
    # Emitted unconditionally so a reader never branches on whether the catalogue
    # resolver is switched on; null stays the ordinary answer even once it is.
    result = PrescriptionFields(medicines=[medicine()], prescribed_date=None, prescriber=None)
    payload = _build_payload(result, [medicine()], [])
    assert payload["fields"]["medicines"][0]["medicine_id"] is None
