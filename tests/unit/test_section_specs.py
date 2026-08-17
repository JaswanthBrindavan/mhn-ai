"""Section specs and the payload the extraction stage stores.

``build_payload`` is the part worth pinning without a database: it decides what actually
lands in ``ai_section_extractions`` — validated fields, ISO dates, and data-quality flags.
"""

import pytest

from app.services.classification import DocumentSection
from app.services.section_extraction import build_payload
from app.services.section_specs import (
    SECTION_SPECS,
    SUPPORTED_SECTIONS,
    BillFields,
    InsuranceFields,
    ScanFields,
    VaccinationFields,
    spec_for,
)


def test_every_spec_is_self_consistent():
    for section, spec in SECTION_SPECS.items():
        assert spec.section is section
        assert spec.json_schema["additionalProperties"] is False
        # Structured output requires every property to be listed as required.
        assert set(spec.json_schema["required"]) == set(spec.json_schema["properties"])
        # Date fields must exist on the model, or normalisation silently does nothing.
        for name in spec.date_fields:
            assert name in spec.model.model_fields
        for earlier, later in spec.date_order:
            assert earlier in spec.date_fields
            assert later in spec.date_fields
        # Same for money: a field listed here but absent from the model would be
        # normalised into a key nothing reads, which no test would otherwise notice.
        for name in spec.amount_fields + spec.currency_fields:
            assert name in spec.model.model_fields


def test_supported_sections_are_the_non_report_ones():
    expected = {
        DocumentSection.INSURANCE,
        DocumentSection.SCANS_IMAGING,
        DocumentSection.VACCINATIONS,
        DocumentSection.BILLS,
    }
    assert expected == SUPPORTED_SECTIONS
    # Reports keep their own deeper pipeline; this package must not claim them.
    assert DocumentSection.REPORTS not in SUPPORTED_SECTIONS


def test_spec_for_an_unhandled_section_names_the_supported_ones():
    with pytest.raises(KeyError, match="insurance"):
        spec_for(DocumentSection.MEDICAL_CONDITION)


def test_insurance_payload_normalises_dates_to_iso():
    spec = spec_for(DocumentSection.INSURANCE)
    fields = InsuranceFields(
        insurer="Star Health",
        start_date="1st October 2019",
        end_date="30/09/2020",
        covered_conditions=[],
        exclusions=[],
    )
    payload = build_payload(spec, fields)

    assert payload["section"] == "insurance"
    assert payload["fields"]["start_date"] == "2019-10-01"
    assert payload["fields"]["end_date"] == "2020-09-30"
    assert payload["flags"] == []


def test_insurance_payload_flags_an_inverted_policy_period():
    """An end before its start is a bad read or a bad document — record it, don't assert it."""
    spec = spec_for(DocumentSection.INSURANCE)
    fields = InsuranceFields(
        start_date="29/07/2027", end_date="27/07/2026", covered_conditions=[], exclusions=[]
    )
    payload = build_payload(spec, fields)

    assert [f["code"] for f in payload["flags"]] == ["dates_out_of_order"]
    assert payload["flags"][0]["field"] == "end_date"
    # The values are still stored — flagged, not discarded.
    assert payload["fields"]["start_date"] == "2027-07-29"


def test_insurance_payload_separates_the_amount_from_its_currency():
    """The shapes a real schedule prints: a rupee sign the text layer lost, Indian
    grouping, and a premium quoted with its symbol attached."""
    spec = spec_for(DocumentSection.INSURANCE)
    fields = InsuranceFields(
        currency="`",
        sum_insured="` 3,00,000.00",
        premium_amount="Rs 52,123.00",
        covered_conditions=[],
        exclusions=[],
    )
    payload = build_payload(spec, fields)["fields"]

    assert payload["currency"] == "INR"
    assert payload["sum_insured"] == "300000.00"
    assert payload["premium_amount"] == "52123.00"


def test_insurance_payload_refuses_a_percentage_as_a_sum():
    """A co-payment share in an amount field is a misread, and storing 20 there would
    present it as twenty rupees."""
    spec = spec_for(DocumentSection.INSURANCE)
    fields = InsuranceFields(sum_insured="20%", covered_conditions=[], exclusions=[])

    assert build_payload(spec, fields)["fields"]["sum_insured"] is None


def test_insurance_payload_keeps_an_absent_policy_absent():
    """The HDFC ERGO schedule prints no co-payment and defers its exclusions to a
    separate policy wording. Nulls and empty lists are the correct answer, and the one
    the liability rests on."""
    spec = spec_for(DocumentSection.INSURANCE)
    fields = InsuranceFields(
        insurer="HDFC ERGO General Insurance Company Limited",
        policy_name="Health Suraksha Policy",
        covered_conditions=[],
        exclusions=[],
    )
    payload = build_payload(spec, fields)["fields"]

    assert payload["co_pay"] is None
    assert payload["exclusions"] == []
    assert payload["currency"] is None
    assert payload["sum_insured"] is None


def test_scan_payload_normalises_a_study_timestamp():
    spec = spec_for(DocumentSection.SCANS_IMAGING)
    fields = ScanFields(
        scan_type="X-Ray",
        body_part="Left Knee",
        scan_date="31/08/2021 18:11:28",
        # An impression, because this fixture was itself an instance of the bug the
        # summary guard now catches: three sentences saying everything looks normal, with
        # no radiologist's read anywhere behind them. See test_scan_summary_guard.py.
        impression="No acute bony injury.",
        summary="An X-ray of the left knee. The report says nothing was broken.",
        findings=[],
    )
    payload = build_payload(spec, fields)

    assert payload["fields"]["scan_date"] == "2021-08-31"
    assert payload["flags"] == []


def test_unreadable_date_becomes_null_rather_than_a_guess():
    spec = spec_for(DocumentSection.SCANS_IMAGING)
    payload = build_payload(spec, ScanFields(scan_date="sometime last winter", findings=[]))
    assert payload["fields"]["scan_date"] is None


def test_vaccination_payload_flags_a_dose_due_before_it_was_given():
    spec = spec_for(DocumentSection.VACCINATIONS)
    fields = VaccinationFields(date_given="02/02/2026", next_due_date="02/02/2025")
    payload = build_payload(spec, fields)

    assert [f["field"] for f in payload["flags"]] == ["next_due_date"]


def test_vaccination_payload_allows_a_missing_next_dose():
    """A completed series has no next dose; that is not a flag."""
    spec = spec_for(DocumentSection.VACCINATIONS)
    payload = build_payload(spec, VaccinationFields(date_given="23/12/2021"))

    assert payload["fields"]["date_given"] == "2021-12-23"
    assert payload["fields"]["next_due_date"] is None
    assert payload["flags"] == []


def test_bill_payload_normalises_the_money_and_the_date():
    spec = spec_for(DocumentSection.BILLS)
    fields = BillFields(
        facility="Apollo Hospitals",
        bill_number="INV/2026/4471",
        bill_date="14th March 2026",
        currency="`",  # a rupee sign the text layer could not resolve
        total_amount="` 1,450.00",
        amount_due="3,00,000",  # Indian grouping
    )
    payload = build_payload(spec, fields)

    assert payload["section"] == "bills"
    assert payload["fields"]["bill_date"] == "2026-03-14"
    assert payload["fields"]["currency"] == "INR"
    assert payload["fields"]["total_amount"] == "1450.00"
    assert payload["fields"]["amount_due"] == "300000"
    assert payload["flags"] == []


def test_bill_payload_refuses_a_percentage_in_an_amount_field():
    """A share, not a sum. Keeping the number would show a discount as the bill total."""
    spec = spec_for(DocumentSection.BILLS)
    payload = build_payload(spec, BillFields(total_amount="20%"))
    assert payload["fields"]["total_amount"] is None
