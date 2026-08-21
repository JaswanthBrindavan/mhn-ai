"""The name is read by the classifier, not the extractor.

Classification is the only stage every document runs, it already sends the first two
pages, and it runs before filing — which is the whole point, since the gate must stop a
document before it enters a wallet.
"""

from app.services.classification import (
    CLASSIFICATION_JSON_SCHEMA,
    SYSTEM_PROMPT,
    DocumentClassification,
)


def test_patient_name_is_parsed() -> None:
    result = DocumentClassification.model_validate_json(
        '{"section":"reports","title":"CBC","handwriting":"none","confidence":0.9,'
        '"reasoning":"","patient_name":"MR RAJESH SHARMA"}'
    )
    assert result.patient_name == "MR RAJESH SHARMA"


def test_patient_name_absent_is_none_not_a_failure() -> None:
    """A document with no printed name must classify normally."""
    result = DocumentClassification.model_validate_json(
        '{"section":"reports","title":"CBC","handwriting":"none","confidence":0.9,'
        '"reasoning":"","patient_name":null}'
    )
    assert result.patient_name is None


def test_blank_name_normalises_to_none() -> None:
    result = DocumentClassification.model_validate_json(
        '{"section":"bills","title":"Bill","handwriting":"none","confidence":0.9,'
        '"reasoning":"","patient_name":"   "}'
    )
    assert result.patient_name is None


def test_schema_requires_patient_name() -> None:
    """Every field is required in structured output; nullable is expressed as a union."""
    assert "patient_name" in CLASSIFICATION_JSON_SCHEMA["required"]
    assert CLASSIFICATION_JSON_SCHEMA["properties"]["patient_name"] == {"type": ["string", "null"]}


def test_prompt_forbids_inventing_a_name() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "patient_name" in lowered
    assert "null" in lowered
