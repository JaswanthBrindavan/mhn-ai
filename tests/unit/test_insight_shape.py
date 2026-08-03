"""The four-part insight contract.

An insight is deliberately not one prose blob: `reports.content` consumers render the
parts separately, so a payload missing one is a broken result, not a terse one. Pydantic
is what enforces that — the model is never asked to repair its own output.
"""

import pytest
from pydantic import ValidationError

from app.services.insights import (
    INSIGHTS_JSON_SCHEMA,
    INSIGHTS_MAX_TOKENS,
    Insight,
)

PARTS = ("explanation", "risk_patterns", "suggestion_heading", "suggestions")


def _valid() -> dict[str, object]:
    return {
        "heading": "Low Hemoglobin - Anaemia Risk",
        "explanation": "This checks the part of your blood that carries oxygen. It "
        "drops when you are low on iron or losing blood.",
        "risk_patterns": "Your haemoglobin is 11.2, below the normal floor of 13.0. "
        "This can leave you tired and short of breath.",
        "suggestion_heading": "Investigate Low Hemoglobin",
        "suggestions": "Get an iron test (ferritin). Eat more leafy greens, beans and "
        "red meat. Recheck the blood count in 8 to 12 weeks.",
        "related_tests": ["HEMOGLOBIN"],
    }


def test_a_complete_insight_validates() -> None:
    insight = Insight.model_validate(_valid())
    assert insight.related_tests == ["HEMOGLOBIN"]


@pytest.mark.parametrize("part", PARTS)
def test_every_explanatory_part_is_required(part: str) -> None:
    """Dropping any one part must fail, not silently produce a half-rendered insight."""
    payload = _valid()
    del payload[part]
    with pytest.raises(ValidationError):
        Insight.model_validate(payload)


@pytest.mark.parametrize("part", PARTS)
def test_an_empty_part_is_rejected(part: str) -> None:
    """An empty string satisfies "present" but renders as a blank section."""
    payload = _valid() | {part: ""}
    with pytest.raises(ValidationError):
        Insight.model_validate(payload)


def test_the_json_schema_matches_the_model() -> None:
    """The hand-written schema is what the model is constrained by; if it drifts from the
    Pydantic model, output validates at the provider and then fails here."""
    assert set(INSIGHTS_JSON_SCHEMA["properties"]["insights"]["items"]["required"]) == set(
        Insight.model_fields
    )


def test_token_ceiling_leaves_room_for_the_four_part_shape() -> None:
    """Four parts per insight is roughly 3x the old single-body output. A 9-insight
    report measured 2,774 output tokens; the old 4096 ceiling would have truncated it,
    and truncation is not detected — it fails validation and burns three paid retries."""
    assert INSIGHTS_MAX_TOKENS >= 8192
