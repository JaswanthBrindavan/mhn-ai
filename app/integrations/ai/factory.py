"""Construct the configured AI provider(s).

The single place AI clients are built, mirroring the boto3 factory. Models and keys come
from settings; an empty model falls back to that provider's mandated default.

**Per-stage providers.** Each stage can run on a different provider, because the stages ask
for very different things:

* *classifying* — pick one label from the first two pages. A small model's job.
* *extracting* — transcribe every row of a table. Mechanical, but the most expensive stage,
  since the whole document goes to the model.
* *generating_insights* — reason about health in plain language for a patient to read. This
  is where a stronger model earns its cost, and it deliberately stays on Claude.

An empty override means "same provider as everything else", which is the default and keeps
single-provider deploys unchanged.
"""

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.config import Settings, get_settings
from app.integrations.ai.anthropic_provider import DEFAULT_MODEL, AnthropicProvider
from app.integrations.ai.base import AIProvider

if TYPE_CHECKING:  # pragma: no cover - typing only
    from google import genai

logger = logging.getLogger(__name__)

#: Stage name (as used in ``ai_process_logs.stage``) -> the settings fields that override it.
_STAGE_OVERRIDES = {
    "classifying": ("classification_provider", "ai_model_classification"),
    "extracting": ("extraction_provider", "ai_model_extraction"),
    # Prescriptions get their own pair rather than sharing the extraction override. The
    # two stages read different things — a lab panel is a printed table, a prescription
    # is often a photograph of handwriting — so a model good enough for one is not
    # automatically right for the other, and pinning them together would mean a change
    # measured on reports silently moves prescriptions too.
    "extracting_prescription": ("prescription_provider", "ai_model_prescription"),
}


@lru_cache
def get_ai_provider() -> AIProvider:
    settings = get_settings()
    # Imported lazily so importing this module (and the app) does not require the
    # anthropic package at import time in environments that never call the model.
    import anthropic

    model = settings.ai_model
    if not model:
        # Say so. Which model runs is a cost decision, and an unset variable silently
        # choosing one is how a deploy ends up paying a different bill than the one the
        # measurements in docs/ai-provider-comparison.md were taken against.
        logger.warning(
            "ai_model_not_set_using_default",
            extra={"model": DEFAULT_MODEL},
        )
        model = DEFAULT_MODEL

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    return AnthropicProvider(client, model)


@lru_cache
def _gemini_client(api_key: str) -> "genai.Client":
    from google import genai

    return genai.Client(api_key=api_key)


def get_gemini_provider(api_key: str, model: str = "") -> AIProvider:
    """Build a Gemini provider from an explicit key and model.

    The key is a parameter rather than read from global settings so the caller's ``Settings``
    is the single source of truth — otherwise the missing-key guard below can pass in a
    process whose environment happens to have a key, which is exactly the situation where it
    most needs to fire.
    """
    from app.integrations.ai.gemini_provider import DEFAULT_MODEL as GEMINI_DEFAULT
    from app.integrations.ai.gemini_provider import GeminiProvider

    if not api_key:
        raise RuntimeError(
            "A stage is configured to use gemini but GOOGLE_API_KEY is not set. Refusing to "
            "run a stage that cannot authenticate."
        )
    return GeminiProvider(_gemini_client(api_key), model or GEMINI_DEFAULT)


def get_stage_provider(settings: Settings, default: AIProvider, *, stage: str) -> AIProvider:
    """The provider a stage should use: its override if configured, else ``default``.

    ``default`` is the provider already injected into the stage, so the ordinary path stays
    dependency-injected and tests keep control by passing their own. Only an explicit
    per-stage override diverges from it.
    """
    provider_field, model_field = _STAGE_OVERRIDES.get(stage, ("", ""))
    if not provider_field:
        return default
    configured = getattr(settings, provider_field).strip().lower()
    if configured == "gemini":
        return get_gemini_provider(
            settings.google_api_key.strip(), getattr(settings, model_field).strip()
        )
    if configured:
        # Falling back is the safe direction — an unrecognised vendor must not receive a
        # document — but silently is not: a typo in the variable reads as "the override is
        # live" while every call goes to the default provider.
        logger.warning(
            "unrecognised_stage_provider_ignored",
            extra={"stage": stage, "setting": provider_field, "value": configured},
        )
    return default
