"""The handwriting verdict: the one classification field that changes routing.

`mostly` means a prescription is filed and never read. That is a refusal to extract, so
the value has to be trustworthy in one direction specifically — an unreadable answer must
degrade to "printed" (extract, as before this field existed), never to "handwritten"
(silently stop reading documents we used to read).
"""

import pytest

from app.services.classification import HANDWRITING_LEVELS, DocumentClassification


def classify(**overrides: object) -> DocumentClassification:
    payload: dict[str, object] = {
        "section": "prescriptions",
        "title": "Medical Prescription",
        "confidence": 0.9,
        "reasoning": "A medicines table with dose and duration.",
    }
    payload.update(overrides)
    return DocumentClassification.model_validate(payload)


@pytest.mark.parametrize("level", HANDWRITING_LEVELS)
def test_the_three_levels_survive_untouched(level: str) -> None:
    assert classify(handwriting=level).handwriting == level


@pytest.mark.parametrize("value", ["MOSTLY", " Mostly ", "None"])
def test_case_and_padding_are_normalised(value: str) -> None:
    assert classify(handwriting=value).handwriting == value.strip().lower()


@pytest.mark.parametrize("value", [None, "", "handwritten", "yes", "unknown", 3, True])
def test_anything_else_degrades_to_printed(value: object) -> None:
    """The safe direction, deliberately.

    "none" means extract — which is exactly what happened before this field existed, so an
    unreadable answer costs nothing. Degrading to "mostly" instead would silently stop
    reading prescriptions the pipeline handles correctly today, and nothing downstream
    would show that it had happened.
    """
    assert classify(handwriting=value).handwriting == "none"


def test_a_missing_field_is_printed() -> None:
    # An older payload, or a provider that dropped the key: same answer, same reason.
    assert classify().handwriting == "none"


def test_an_unreadable_handwriting_value_does_not_lose_the_classification() -> None:
    """It is one advisory field on a document that was otherwise classified correctly.

    Failing validation here would throw away the section, the title and the paid model
    call over a word the model got wrong.
    """
    result = classify(handwriting="scribbled")
    assert result.section.value == "prescriptions"
    assert result.title == "Medical Prescription"
