"""Construct the configured AI provider.

The single place an Anthropic client is built, mirroring the boto3 factory. Model and
key come from settings; an empty model falls back to the mandated default.
"""

from functools import lru_cache

from app.core.config import get_settings
from app.integrations.ai.anthropic_provider import DEFAULT_MODEL, AnthropicProvider
from app.integrations.ai.base import AIProvider


@lru_cache
def get_ai_provider() -> AIProvider:
    settings = get_settings()
    # Imported lazily so importing this module (and the app) does not require the
    # anthropic package at import time in environments that never call the model.
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    return AnthropicProvider(client, settings.ai_model or DEFAULT_MODEL)
