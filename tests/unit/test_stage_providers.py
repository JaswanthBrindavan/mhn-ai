"""Per-stage provider selection.

Each stage can be pointed at a different provider, because they ask for different things:
classification picks a label, extraction transcribes a table, insight generation reasons
about health. Insights has **no** override on purpose — that stays on Claude.

The important property is that an unconfigured stage returns the *injected* provider, so the
ordinary path stays dependency-injected and tests keep control.
"""

import pytest

from app.core.config import Settings
from app.integrations.ai.factory import get_stage_provider


class _Sentinel:
    """Stands in for the provider a stage was handed by DI."""


def _settings(**over) -> Settings:
    base = {
        "database_url": "postgresql+psycopg://x/y",
        "mhn_service_token": "t" * 32,
    }
    return Settings(**{**base, **over})


@pytest.mark.parametrize(
    "stage",
    ["classifying", "extracting", "extracting_section", "generating_insights"],
)
def test_no_override_returns_the_injected_provider(stage):
    injected = _Sentinel()
    assert get_stage_provider(_settings(), injected, stage=stage) is injected


def test_insights_has_no_override_even_when_others_are_set():
    """Insights is the patient-facing reasoning stage; it must not be swappable by config."""
    injected = _Sentinel()
    settings = _settings(
        classification_provider="gemini",
        extraction_provider="gemini",
        section_provider="gemini",
        prescription_provider="gemini",
    )
    assert get_stage_provider(settings, injected, stage="generating_insights") is injected


def test_section_extraction_is_redirectable_on_purpose():
    """It calls ``get_stage_provider`` rather than ``ctx.ai``, and that is deliberate now.

    It used to call ``ctx.ai`` directly, which made it the one transcription stage no
    configuration could move — insulation by accident. That accident mattered more once the
    whole document started going to the model, so the override exists and is named.
    """
    settings = _settings(section_provider="gemini", google_api_key="")
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        get_stage_provider(settings, _Sentinel(), stage="extracting_section")


def test_an_unknown_stage_falls_back_to_the_injected_provider():
    injected = _Sentinel()
    assert get_stage_provider(_settings(), injected, stage="something_new") is injected


@pytest.mark.parametrize(
    ("stage", "field"),
    [("classifying", "classification_provider"), ("extracting", "extraction_provider")],
)
def test_gemini_override_without_a_key_fails_loudly(stage, field):
    """Better to refuse at the stage than to run one that cannot authenticate."""
    settings = _settings(**{field: "gemini"}, google_api_key="")
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        get_stage_provider(settings, _Sentinel(), stage=stage)


@pytest.mark.parametrize("value", ["anthropic", "claude", "", "  "])
def test_only_gemini_is_recognised_as_an_override(value):
    injected = _Sentinel()
    settings = _settings(extraction_provider=value)
    assert get_stage_provider(settings, injected, stage="extracting") is injected
