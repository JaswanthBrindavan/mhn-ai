"""The two checks that stand between a model and a stored drug name.

Neither calls the API. What is being tested is not whether the model reads a page — it is
what happens to what it returns, which is where a wrong answer becomes a stored fact.
"""

from app.services.prescriptions import (
    PrescribedMedicine,
    PrescriptionFields,
    _appears_in,
    _build_payload,
    _has_drug_identity,
    _normalise,
)

#: A two-column EMR prescription as the text extractor actually returns it: the name is
#: split across the column boundary, with the dosing column in between its halves.
PAGE = _normalise(
    "2) DAPAGLIFLOZIN-TABLET-5MG-   Once Daily ( 0 - 1 - 0 - 0 ) Tablet Orally\n"
    "   DAPEFY                      After Food For 3 Months | Qty: 90\n"
    "3) Tab. DOLO-650               twice daily\n"
)


def medicine(**kwargs: str) -> PrescribedMedicine:
    base = {"name_as_written": "Tab. DOLO 650", "name_clean": "DOLO"}
    base.update(kwargs)
    return PrescribedMedicine(**base)  # type: ignore[arg-type]


# --- is it on the page? -----------------------------------------------------


def test_a_name_split_across_columns_is_still_found() -> None:
    """The whole run is never contiguous in the extracted text, so parts have to match.

    Split on whitespace this is one token and matches nothing, and a real medicine is
    dropped. Split on the hyphens, its drug parts are found individually.
    """
    assert _appears_in("DAPAGLIFLOZIN-TABLET-5MG-DAPEFY", PAGE)


def test_tidying_by_the_reader_is_not_mistaken_for_invention() -> None:
    assert _appears_in("TAB DOLO 650", PAGE)  # punctuation dropped
    assert _appears_in("DOLO-650 Tablet", PAGE)  # dosage form supplied
    assert _appears_in("DAPEFY", PAGE)  # brand on its own


def test_a_drug_the_page_never_mentions_is_rejected() -> None:
    assert not _appears_in("Warfarin", PAGE)
    assert not _appears_in("Metformin 500", PAGE)


def test_an_invented_name_in_the_documents_own_style_is_still_rejected() -> None:
    """Matching on parts must not become a way to smuggle a name through.

    This is shaped exactly like the real EMR names on the page — same hyphen-joined
    generic/form/strength/brand run — and none of its drug parts are there.
    """
    assert not _appears_in("WARFARIN-TABLET-5MG-SOFARIN", PAGE)


# --- does it name a drug at all? --------------------------------------------


def test_a_name_of_only_dosage_forms_is_not_a_medicine() -> None:
    """"Tablet" is printed on nearly every prescription, so it passes "is it on the page"
    while naming nothing. That is a misread heading, not a hallucination, and it needs a
    different check to catch it."""
    assert _appears_in("Tablet", PAGE)  # it IS on the page
    assert not _has_drug_identity("Tablet")  # but it names no drug
    assert not _has_drug_identity("Cap")
    assert not _has_drug_identity("e/d")


def test_a_real_name_has_identity() -> None:
    assert _has_drug_identity("DAPEFY")
    assert _has_drug_identity("Dolo 650")
    assert _has_drug_identity("DAPAGLIFLOZIN-TABLET-5MG-DAPEFY")


# --- the stored payload -----------------------------------------------------


def test_payload_carries_the_normalised_schedule() -> None:
    result = PrescriptionFields(
        medicines=[medicine(frequency_raw="1-0-1 after food")],
        prescribed_date=None,
        prescriber=None,
    )
    payload = _build_payload(result, list(result.medicines), [])
    stored = payload["fields"]["medicines"][0]
    assert stored["frequency_normalized"]["morning"] == 1.0
    assert stored["frequency_normalized"]["night"] == 1.0
    assert stored["frequency_normalized"]["with_food"] is True
    assert payload["section"] == "prescriptions"
    assert payload["flags"] == []


def test_the_lookup_table_beats_the_model_on_a_form_it_knows() -> None:
    """"Tab." is Tablet in every document ever printed, so it is never put to a model that
    could answer differently tomorrow."""
    result = PrescriptionFields(
        medicines=[medicine(form_raw="Tab.", form="Injection")],
        prescribed_date=None,
        prescriber=None,
    )
    payload = _build_payload(result, list(result.medicines), [])
    assert payload["fields"]["medicines"][0]["form_normalized"] == "Tablet"


def test_the_model_settles_what_the_table_cannot() -> None:
    """A gel is Cream or Ointment depending on its base — a property of the product, not
    of the word, so no lookup table can decide it."""
    result = PrescriptionFields(
        medicines=[
            PrescribedMedicine(
                name_as_written="ERYTOP-N GEL 15 GM",
                name_clean="ERYTOP-N",
                form_raw="GEL",
                form="Cream",
            )
        ],
        prescribed_date=None,
        prescriber=None,
    )
    payload = _build_payload(result, list(result.medicines), [])
    assert payload["fields"]["medicines"][0]["form_normalized"] == "Cream"


def test_a_printed_form_beats_a_token_scraped_from_the_name() -> None:
    """``form_raw`` is what the document said the form was; the name is a last resort.

    Letting a stray "Tab." in a product name override the model's reading of the actual
    form column would report a gel as a tablet.
    """
    result = PrescriptionFields(
        medicines=[
            PrescribedMedicine(
                name_as_written="Tab-brand GEL 15 GM",
                name_clean="Tab-brand",
                form_raw="GEL",
                form="Ointment",
            )
        ],
        prescribed_date=None,
        prescriber=None,
    )
    payload = _build_payload(result, list(result.medicines), [])
    assert payload["fields"]["medicines"][0]["form_normalized"] == "Ointment"


def test_the_model_may_not_supply_a_form_the_document_never_printed() -> None:
    """Every model knows paracetamol comes as a tablet. That is not a reason to record it.

    With no ``form_raw`` there was nothing printed to classify, so the model's answer is
    discarded — a form nobody wrote down is indistinguishable, once stored, from one the
    prescriber did.
    """
    result = PrescriptionFields(
        medicines=[
            PrescribedMedicine(
                name_as_written="Paracetamol 500 mg",
                name_clean="Paracetamol",
                form_raw=None,
                form="Tablet",
            )
        ],
        prescribed_date=None,
        prescriber=None,
    )
    payload = _build_payload(result, list(result.medicines), [])
    assert payload["fields"]["medicines"][0]["form_normalized"] is None


def test_a_form_outside_the_nine_is_discarded_on_the_way_in() -> None:
    """The schema cannot express "one of nine, or nothing", so the validator is the
    guarantee rather than a second opinion."""
    assert PrescribedMedicine(
        name_as_written="X", name_clean="X", form="Suppository"
    ).form is None
    assert PrescribedMedicine(name_as_written="X", name_clean="X", form="Cream").form == "Cream"


def test_a_rejected_name_is_flagged_not_silently_dropped() -> None:
    result = PrescriptionFields(medicines=[], prescribed_date=None, prescriber=None)
    payload = _build_payload(result, [], ["Warfarin 5mg"])
    assert payload["flags"] == [
        {"code": "names_not_on_document", "names": ["Warfarin 5mg"]}
    ]


def test_an_unverified_read_says_so() -> None:
    """An OCR'd page cannot reject, so the payload must not look like it passed a check
    it never had."""
    result = PrescriptionFields(medicines=[medicine()], prescribed_date=None, prescriber=None)
    payload = _build_payload(result, list(result.medicines), [], verified=False)
    assert {f["code"] for f in payload["flags"]} == {"names_unverified"}


def test_an_unnormalisable_frequency_is_flagged_and_kept_raw() -> None:
    result = PrescriptionFields(
        medicines=[medicine(frequency_raw="Alternate day")],
        prescribed_date=None,
        prescriber=None,
    )
    payload = _build_payload(result, list(result.medicines), [])
    stored = payload["fields"]["medicines"][0]
    assert stored["frequency_normalized"] is None
    assert stored["frequency_raw"] == "Alternate day"
    assert {f["code"] for f in payload["flags"]} == {"frequency_not_normalized"}
