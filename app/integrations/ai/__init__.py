"""AI provider integration.

The provider is an interface (`AIProvider`) so the Anthropic client can be swapped
or, in tests, faked — the real API call is neither free nor deterministic, so every
piece of logic around it (validation, persistence, cost, reject rules) is exercised
against a fake, and only the thin request-building method talks to Anthropic.
"""

from app.integrations.ai.base import (
    AIProvider,
    AIProviderError,
    AIUsage,
    DocumentPayload,
    StructuredResponse,
)

__all__ = [
    "AIProvider",
    "AIProviderError",
    "AIUsage",
    "DocumentPayload",
    "StructuredResponse",
]
