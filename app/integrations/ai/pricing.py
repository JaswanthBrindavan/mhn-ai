"""Model pricing and cost estimation.

Cost is computed by application code from token counts, never asked of the model.
Prices are USD per 1,000,000 tokens and live here as the single place to update when
they change. Cache reads bill at ~0.1x input, cache writes at ~1.25x input (5-minute
TTL) — the standard Anthropic multipliers.
"""

import logging
import re
from dataclasses import dataclass
from decimal import Decimal

from app.integrations.ai.base import AIUsage

logger = logging.getLogger(__name__)

_PER_MILLION = Decimal(1_000_000)
_CACHE_READ_MULTIPLIER = Decimal("0.1")
_CACHE_WRITE_MULTIPLIER = Decimal("1.25")
# The API resolves an alias to a dated snapshot id (e.g. "claude-haiku-4-5" ->
# "claude-haiku-4-5-20251001"), and response.model carries the dated form. The price
# table is keyed by alias, so strip a trailing -YYYYMMDD before falling back.
_DATE_SUFFIX = re.compile(r"-\d{8}$")


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: Decimal
    output_per_million: Decimal


#: Keyed by exact model id. Extend as models are adopted. An unpriced model costs 0 and
#: is WARNED about here — the comment used to say the caller flagged it, and no caller
#: ever did. That silence is exactly how Haiku's real cost stayed hidden until PR #7: the
#: table was keyed by alias while the API returned a dated snapshot, so every Haiku stage
#: logged $0 and the money column looked fine.
PRICES: dict[str, ModelPrice] = {
    "claude-opus-4-8": ModelPrice(Decimal("5.00"), Decimal("25.00")),
    "claude-opus-4-7": ModelPrice(Decimal("5.00"), Decimal("25.00")),
    "claude-sonnet-5": ModelPrice(Decimal("3.00"), Decimal("15.00")),
    "claude-sonnet-4-6": ModelPrice(Decimal("3.00"), Decimal("15.00")),
    "claude-haiku-4-5": ModelPrice(Decimal("1.00"), Decimal("5.00")),
    # Gemini (ai.google.dev/gemini-api/docs/pricing, standard tier, 2026-07-29).
    "gemini-3.1-flash-lite": ModelPrice(Decimal("0.25"), Decimal("1.50")),
    "gemini-3.5-flash-lite": ModelPrice(Decimal("0.30"), Decimal("2.50")),
    "gemini-3.1-flash": ModelPrice(Decimal("0.50"), Decimal("3.00")),
}


def estimate_cost_usd(model: str, usage: AIUsage) -> Decimal:
    """Best-effort USD cost for one call. Returns 0 for an unpriced model, and says so."""
    # Exact match first; then retry with any dated snapshot suffix stripped.
    price = PRICES.get(model) or PRICES.get(_DATE_SUFFIX.sub("", model))
    if price is None:
        # A zero in the cost column is indistinguishable from a stage that made no call,
        # so an unpriced model quietly understates spend across every document it touches.
        # Warn rather than raise: a missing price must never fail a document that has
        # already been paid for and processed. (A stage that made no call never reaches
        # here — log_process skips the estimate when there is no usage to price.)
        logger.warning("model_not_priced", extra={"model": model})
        return Decimal("0")

    input_rate = price.input_per_million
    cost = (
        Decimal(usage.input_tokens) * input_rate
        + Decimal(usage.output_tokens) * price.output_per_million
        + Decimal(usage.cache_read_input_tokens) * input_rate * _CACHE_READ_MULTIPLIER
        + Decimal(usage.cache_creation_input_tokens) * input_rate * _CACHE_WRITE_MULTIPLIER
    )
    return (cost / _PER_MILLION).quantize(Decimal("0.000001"))
