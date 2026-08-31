"""The four-part insight contract.

An insight is deliberately not one prose blob: `reports.content` consumers render the
parts separately, so a payload missing one is a broken result, not a terse one. Pydantic
is what enforces that — the model is never asked to repair its own output.
"""

import pytest
from pydantic import ValidationError

from app.services.insights import (
    FIELD_LIMITS,
    INSIGHTS_JSON_SCHEMA,
    INSIGHTS_MAX_TOKENS,
    SYSTEM_PROMPT,
    DocumentInsights,
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


def _pydantic_cap(field: str) -> int:
    """The validator's ceiling for one field.

    `summary` is a property of the whole payload rather than of one insight, which is why
    it lives on a different model — and it is the field that most recently blew its cap,
    so leaving it out of these checks would miss the case they were written for.

    The constraints arrive as a list of annotated-types markers (MinLen, MaxLen) in
    declaration order, so this looks one up by attribute rather than by position — which
    would silently read min_length the day someone reorders them.
    """
    model = DocumentInsights if field == "summary" else Insight
    return next(
        m.max_length for m in model.model_fields[field].metadata if hasattr(m, "max_length")
    )


def test_the_model_is_told_every_limit_it_is_validated_against() -> None:
    """The bug behind three thrown-away payloads in one day.

    A character limit existed only in the Pydantic model; the prompt asked for a WORD
    budget. Output was rejected for breaking a rule that had never been stated in the unit
    it was measured in — and because a validation failure here is TRANSIENT, that cost
    three paid retries and then a report with no insights at all. `summary` came back at
    779 against 700.

    The prompt is the ONLY place this can be said. Anthropic's structured outputs do not
    support `maxLength`, so it cannot be put in the schema and enforced — see
    FIELD_LIMITS. Change a limit and this test fails until the prompt says so too.
    """
    for field, limit in FIELD_LIMITS.items():
        assert f"{limit} CHARACTERS" in SYSTEM_PROMPT, (
            f"{field} is validated against {limit} characters and the prompt never says "
            f"so. The model cannot be constrained to a length — only asked — so a limit "
            f"missing from the prompt is a limit that exists nowhere the model can see."
        )


def test_the_schema_claims_no_length_it_cannot_enforce() -> None:
    """`maxLength` in this schema would be silently stripped by the SDK before sending.

    It reads as a guarantee and is not one, which is worse than its absence: someone would
    then tighten the validator towards it, on the theory that generation is constrained.
    Tried on 2026-08-31, caught before it shipped.
    """
    item = INSIGHTS_JSON_SCHEMA["properties"]["insights"]["items"]["properties"]
    for node in (*item.values(), INSIGHTS_JSON_SCHEMA["properties"]["summary"]):
        assert "maxLength" not in node
        assert "minLength" not in node


def test_the_validator_sits_well_above_the_limit_the_model_is_given() -> None:
    """The two numbers do different jobs, and conflating them is what kept breaking.

    `FIELD_LIMITS` is the product decision and reaches the model. The Pydantic cap only
    decides whether to throw the answer away — so it belongs where that is worth doing: an
    answer twice as long as asked is broken, one slightly over is just long.
    """
    for field, stated in FIELD_LIMITS.items():
        assert _pydantic_cap(field) > stated, (
            f"{field}'s validator cap is not above the limit the model is given, so an "
            "ordinary overshoot destroys the payload rather than merely reading long."
        )


def test_token_ceiling_leaves_room_for_the_four_part_shape() -> None:
    """Four parts per insight is roughly 3x the old single-body output. A 9-insight
    report measured 2,774 output tokens; the old 4096 ceiling would have truncated it,
    and truncation is not detected — it fails validation and burns three paid retries."""
    assert INSIGHTS_MAX_TOKENS >= 8192
