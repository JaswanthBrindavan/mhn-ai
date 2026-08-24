"""The dates are read by the classifier, and Python decides which one counts.

Classification is the only stage every document runs and the only one that finishes
before the user is asked to confirm what they uploaded — section extraction reads richer
dates, but it runs after the point where the answer is needed, and it is the stage
on-demand analysis defers. So the date is transcribed here, like the patient name.

What the model is asked for is every labelled date, verbatim. Which of them is *the*
date is `app/services/document_date.py`'s decision, tested separately.
"""

from datetime import date

from app.services.classification import (
    CLASSIFICATION_JSON_SCHEMA,
    SYSTEM_PROMPT,
    DocumentClassification,
    chosen_date,
)

_BASE = (
    '"section":"reports","title":"CBC","handwriting":"none","confidence":0.9,'
    '"reasoning":"","patient_name":null'
)


def test_labelled_dates_are_parsed() -> None:
    result = DocumentClassification.model_validate_json(
        "{" + _BASE + ',"dates":[{"label":"Sample Collected","value":"12/03/2026"},'
        '{"label":"Reported On","value":"15/03/2026"}]}'
    )
    assert [(d.label, d.value) for d in result.dates] == [
        ("Sample Collected", "12/03/2026"),
        ("Reported On", "15/03/2026"),
    ]


def test_absent_dates_are_an_empty_list_not_a_failure() -> None:
    """Every classification written before this field existed must still parse."""
    result = DocumentClassification.model_validate_json("{" + _BASE + "}")
    assert result.dates == []


def test_an_empty_list_is_a_real_answer() -> None:
    # Many bare X-rays, vaccination cards and bills print no date at all. That is
    # different from not having looked, and the app renders the two differently.
    result = DocumentClassification.model_validate_json("{" + _BASE + ',"dates":[]}')
    assert result.dates == []


def test_a_null_label_becomes_an_empty_one() -> None:
    # The schema allows a null label because a date can be printed bare in a header.
    result = DocumentClassification.model_validate_json(
        "{" + _BASE + ',"dates":[{"label":null,"value":"12/03/2026"}]}'
    )
    assert result.dates[0].label == ""


def test_malformed_dates_degrade_to_empty_rather_than_failing_the_document() -> None:
    # An advisory field must never cost a document its classification — the rule
    # `handwriting` and `patient_name` already follow. Failing here would lose the
    # section, the title and the patient name over a date.
    result = DocumentClassification.model_validate_json("{" + _BASE + ',"dates":"not a list"}')
    assert result.dates == []


def test_entries_with_no_value_are_dropped_not_defaulted() -> None:
    # An empty value would parse to None downstream and read as a date we considered
    # and rejected, rather than one that was never there.
    result = DocumentClassification.model_validate_json(
        "{" + _BASE + ',"dates":[{"label":"Invoice Date","value":""},'
        '{"label":"Bill Date","value":"01/04/2026"}]}'
    )
    assert [d.label for d in result.dates] == ["Bill Date"]


def test_a_pathological_list_is_bounded() -> None:
    entries = ",".join(f'{{"label":"D{i}","value":"01/04/2026"}}' for i in range(40))
    result = DocumentClassification.model_validate_json("{" + _BASE + ',"dates":[' + entries + "]}")
    assert len(result.dates) == 20


def test_chosen_date_applies_the_section_rule() -> None:
    result = DocumentClassification.model_validate_json(
        "{" + _BASE + ',"dates":[{"label":"Reported On","value":"15/03/2026"},'
        '{"label":"Sample Collected","value":"12/03/2026"}]}'
    )
    assert chosen_date(result) == (date(2026, 3, 12), "Sample Collected")


def test_chosen_date_of_a_dateless_document() -> None:
    result = DocumentClassification.model_validate_json("{" + _BASE + "}")
    assert chosen_date(result) == (None, None)


def test_schema_requires_dates_and_closes_the_entries() -> None:
    """Every field is required in structured output; nullable is a union."""
    assert "dates" in CLASSIFICATION_JSON_SCHEMA["required"]
    prop = CLASSIFICATION_JSON_SCHEMA["properties"]["dates"]
    assert prop["type"] == "array"
    assert prop["items"]["additionalProperties"] is False
    assert prop["items"]["required"] == ["label", "value"]
    assert prop["items"]["properties"]["label"] == {"type": ["string", "null"]}
    assert prop["items"]["properties"]["value"] == {"type": "string"}


def test_prompt_asks_for_every_date_and_forbids_inventing_one() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "dates" in lowered
    # The two halves that matter: all of them, and none made up.
    assert "every date" in lowered
    assert "never infer" in lowered
