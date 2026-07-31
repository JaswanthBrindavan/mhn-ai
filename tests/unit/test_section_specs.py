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


def test_supported_sections_are_the_non_report_ones():
    expected = {
        DocumentSection.INSURANCE,
        DocumentSection.SCANS_IMAGING,
        DocumentSection.VACCINATIONS,
    }
    assert expected == SUPPORTED_SECTIONS
    # Reports keep their own deeper pipeline; this package must not claim them.
    assert DocumentSection.REPORTS not in SUPPORTED_SECTIONS


def test_spec_for_an_unhandled_section_names_the_supported_ones():
    with pytest.raises(KeyError, match="insurance"):
        spec_for(DocumentSection.BILLS)


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


def test_scan_payload_normalises_a_study_timestamp():
    spec = spec_for(DocumentSection.SCANS_IMAGING)
    fields = ScanFields(
        scan_type="X-Ray",
        body_part="Left Knee",
        scan_date="31/08/2021 18:11:28",
        summary="An X-ray of the left knee. Everything looks normal.",
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
