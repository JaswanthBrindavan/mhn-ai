"""Cost/provenance logging for AI stages, shared so every stage logs identically.

One ``ai_process_logs`` row per ``(run_item_id, stage, attempt)``: a re-run of the SAME
attempt (SQS redelivery) updates it, so a single attempt's cost is never double-counted;
a genuine retry is a new attempt and gets its own row. This is the money path — it lives
in one place on purpose.
"""

import time
from decimal import Decimal

from pydantic import ValidationError
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.integrations.ai.base import StructuredResponse
from app.integrations.ai.pricing import estimate_cost_usd
from app.models.ai_results import AiProcessLog
from app.workers.stagetypes import StageContext

#: What the provider and model columns say when a stage completed without calling a model
#: at all. ``generate_insights`` skips the paid call when every result is in range, and
#: naming a model there claimed a call that never happened — in the one table that exists
#: to say what was spent. Both columns are NOT NULL, so this is the sentinel.
NO_CALL = "skipped"


def log_process(
    ctx: StageContext,
    *,
    stage: str,
    prompt_version: str,
    schema_version: str,
    outcome: str,
    duration_ms: int,
    response: StructuredResponse | None = None,
    error_code: str | None = None,
    detail: str | None = None,
) -> None:
    usage = response.usage if response is not None else None
    # Both come from the response or neither does. Falling back to configuration was how
    # a skipped stage came to log Haiku for a call it never made, and a hardcoded provider
    # named Anthropic for every document Gemini actually received.
    model = response.model if response is not None else NO_CALL
    provider = response.provider if response is not None else NO_CALL
    cost = estimate_cost_usd(model, usage) if usage is not None else Decimal("0")

    values = {
        "run_item_id": ctx.item_id,
        "document_id": ctx.document_id,
        "stage": stage,
        "attempt": ctx.attempt,
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "input_tokens": usage.input_tokens if usage else 0,
        "output_tokens": usage.output_tokens if usage else 0,
        "cache_read_input_tokens": usage.cache_read_input_tokens if usage else 0,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens if usage else 0,
        "estimated_cost_usd": cost,
        "duration_ms": duration_ms,
        "outcome": outcome,
        "error_code": error_code,
        "error_detail": detail,
    }
    # The conflict target keys can't change on update, so drop them from the SET.
    update = {k: v for k, v in values.items() if k not in ("run_item_id", "stage", "attempt")}

    stmt = (
        pg_insert(AiProcessLog)
        .values(**values)
        .on_conflict_do_update(constraint="uq_ai_process_logs_item_stage_attempt", set_=update)
    )
    ctx.session.execute(stmt)
    ctx.session.commit()


def sanitize_validation_error(exc: ValidationError) -> str:
    # Field locations and messages only — never the offending model output/value,
    # which could echo report contents.
    parts = [f"{'.'.join(str(p) for p in err['loc'])}: {err['type']}" for err in exc.errors()]
    return "; ".join(parts)[:2000]


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
