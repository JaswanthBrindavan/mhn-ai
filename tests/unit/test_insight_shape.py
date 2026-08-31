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


#: The word budget the prompt states for each capped field. The caps must sit ABOVE these
#: with real headroom — see `_CHARS_PER_WORD` below.
PROMPT_WORD_BUDGETS = {
    "explanation": 40,
    "risk_patterns": 40,
    "suggestions": 60,
    "suggestion_heading": 6,
}

#: Characters of headroom per budgeted word. Not a style preference — the arithmetic of a
#: failure that has already happened once. `risk_patterns` sat at 8.7 and rejected real
#: output in production (`insights.0.risk_patterns: string_too_long`), and because this
#: stage treats a validation failure as TRANSIENT that cost three paid retries and then a
#: report with no insights at all. Medical prose spends characters fast: test names, units
#: and a cited limit are long tokens, and one insight may group several results.
_CHARS_PER_WORD = 11


def test_every_cap_sits_above_the_budget_the_prompt_states() -> None:
    """The rule that was broken, expressed as arithmetic rather than as four numbers.

    The class docstring already says never to tighten a cap without tightening the
    prompt's budget with it. This is the other half: a cap is a SAFETY NET, and the net
    firing is worse than the thing it catches — the reader gets nothing at all, instead of
    one paragraph running long. Tighten a budget here and this test tells you which cap
    has to move with it.
    """
    for field, words in PROMPT_WORD_BUDGETS.items():
        # The constraints come through as a list of annotated-types markers (MinLen,
        # MaxLen), in declaration order — so it is looked up by attribute rather than by
        # position, which would silently read min_length the day someone reorders them.
        cap = next(
            m.max_length for m in Insight.model_fields[field].metadata if hasattr(m, "max_length")
        )
        assert cap >= words * _CHARS_PER_WORD, (
            f"{field} is capped at {cap} against a stated budget of {words} words "
            f"({cap / words:.1f} chars/word). An ordinary overshoot destroys the payload."
        )


def test_token_ceiling_leaves_room_for_the_four_part_shape() -> None:
    """Four parts per insight is roughly 3x the old single-body output. A 9-insight
    report measured 2,774 output tokens; the old 4096 ceiling would have truncated it,
    and truncation is not detected — it fails validation and burns three paid retries."""
    assert INSIGHTS_MAX_TOKENS >= 8192
