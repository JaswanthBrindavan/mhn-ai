"""The two conditions no stage should try to parse a response through.

Both were previously handled per stage, or not at all. Truncation is the one that cost
money: a response cut at the token ceiling fails Pydantic, was recorded as
``invalid_model_output``, treated as transient, and retried at full price to fail
identically every time.
"""

import pytest

from app.integrations.ai.base import AIUsage, StructuredResponse
from app.services.ai_logging import check_response
from app.workers.stagetypes import PermanentStageError, TransientStageError


def _response(stop_reason: str | None) -> StructuredResponse:
    return StructuredResponse(
        text="{}",
        provider="anthropic",
        model="claude-haiku-4-5",
        stop_reason=stop_reason,
        usage=AIUsage(input_tokens=10, output_tokens=5),
    )


def _recorder() -> tuple[list[dict], object]:
    rows: list[dict] = []

    def log(**kwargs: object) -> None:
        rows.append(dict(kwargs))

    return rows, log


def test_a_clean_response_passes_and_logs_nothing():
    rows, log = _recorder()
    check_response(_response("end_turn"), log=log, duration_ms=1, what="extraction")
    assert rows == []


def test_a_refusal_is_transient():
    """The safety classifier is not deterministic, so a retry can legitimately differ."""
    rows, log = _recorder()
    with pytest.raises(TransientStageError, match="refused"):
        check_response(_response("refusal"), log=log, duration_ms=7, what="extraction")

    assert rows[0]["outcome"] == "refused"
    assert rows[0]["error_code"] == "model_refusal"


def test_truncation_is_permanent_and_carries_its_own_code():
    """At temperature=0 a truncated response truncates again. Retrying only spends the
    attempt cap to arrive at the same place, so this ends the item rather than looping."""
    rows, log = _recorder()
    with pytest.raises(PermanentStageError) as excinfo:
        check_response(_response("max_tokens"), log=log, duration_ms=9, what="extraction")

    assert excinfo.value.code == "response_truncated"
    assert rows[0]["outcome"] == "truncated"
    assert rows[0]["error_code"] == "response_truncated"
    # Logged with the response, so the row still records what the failed call cost.
    assert rows[0]["response"].usage.output_tokens == 5


def test_a_refusal_wins_over_truncation():
    """Only one stop reason can be set; this pins the order so the codes cannot swap."""
    rows, log = _recorder()
    with pytest.raises(TransientStageError):
        check_response(_response("refusal"), log=log, duration_ms=1, what="insights")
    assert rows[0]["error_code"] == "model_refusal"
