"""Cost estimation is deterministic application code, not something the model does."""

from decimal import Decimal

from app.integrations.ai.base import AIUsage
from app.integrations.ai.pricing import estimate_cost_usd


def test_opus_cost_from_input_and_output():
    # 1200 input @ $5/1M + 40 output @ $25/1M = 0.006 + 0.001 = 0.007
    usage = AIUsage(input_tokens=1200, output_tokens=40)
    assert estimate_cost_usd("claude-opus-4-8", usage) == Decimal("0.007000")


def test_cache_reads_and_writes_are_priced():
    usage = AIUsage(
        input_tokens=0,
        output_tokens=0,
        cache_read_input_tokens=1_000_000,  # 0.1 x $5 = $0.50
        cache_creation_input_tokens=1_000_000,  # 1.25 x $5 = $6.25
    )
    assert estimate_cost_usd("claude-opus-4-8", usage) == Decimal("6.750000")


def test_dated_snapshot_id_prices_same_as_alias():
    # response.model is the dated snapshot (claude-haiku-4-5-20251001); the price table
    # is keyed by alias. The dated id must price identically, not fall through to $0.
    usage = AIUsage(input_tokens=76_828, output_tokens=87)
    dated = estimate_cost_usd("claude-haiku-4-5-20251001", usage)
    alias = estimate_cost_usd("claude-haiku-4-5", usage)
    assert dated == alias
    assert dated > Decimal("0")


def test_unknown_model_costs_zero_rather_than_guessing():
    assert estimate_cost_usd("some-unlisted-model", AIUsage(1000, 1000)) == Decimal("0")
    # A dated suffix on an unknown alias still resolves to nothing, not a guess.
    assert estimate_cost_usd("mystery-model-20260101", AIUsage(1000, 1000)) == Decimal("0")


def test_zero_usage_is_zero():
    assert estimate_cost_usd("claude-opus-4-8", AIUsage(0, 0)) == Decimal("0.000000")
