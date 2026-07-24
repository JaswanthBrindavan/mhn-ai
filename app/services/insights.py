"""Insight generation — the third pipeline stage (runs after extraction).

Flow: read this item's validated extraction, ask the model for brief informational
insights over that structured data (never the raw file, so it cannot introduce values
that bypassed extraction), validate with Pydantic (never repaired), attach a fixed
disclaimer, then persist to ``ai_report_insights`` and a process log.

Insights are informational only — never a diagnosis, emergency instruction, or medical
certainty. That is enforced by the system prompt; a fixed disclaimer is always stored
alongside them. When there are no extracted results there is nothing to interpret, so the
model call is skipped entirely.

Idempotent: the insights row and the process log are upserted, so a redelivery that
re-runs the stage overwrites its own prior attempt rather than duplicating rows.
"""

import json
import logging
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.integrations.ai.base import AIProviderError
from app.models.ai_results import AiReportExtraction, AiReportInsight
from app.services.ai_logging import elapsed_ms, log_process, sanitize_validation_error
from app.workers.stagetypes import StageContext, TransientStageError

logger = logging.getLogger(__name__)

PROMPT_VERSION = "ins-2026-07-24"
SCHEMA_VERSION = "ins-1"
STAGE_NAME = "generating_insights"
INSIGHTS_MAX_TOKENS = 4096

#: Stored with every insights payload. Informational framing is not left to the model.
DISCLAIMER = (
    "These insights are informational only and are not a medical diagnosis or advice. "
    "Discuss your results with a qualified healthcare professional."
)


class Insight(BaseModel):
    heading: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=2000)
    #: Test names from the extraction this insight refers to.
    related_tests: list[str] = Field(default_factory=list)


class DocumentInsights(BaseModel):
    """Validated model output. An empty list is valid (nothing noteworthy to say)."""

    insights: list[Insight]
    summary: str | None = Field(default=None, max_length=2000)


_INSIGHT_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "heading": {"type": "string"},
        "body": {"type": "string"},
        "related_tests": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["heading", "body", "related_tests"],
    "additionalProperties": False,
}
INSIGHTS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "insights": {"type": "array", "items": _INSIGHT_ITEM_SCHEMA},
        "summary": {"type": ["string", "null"]},
    },
    "required": ["insights", "summary"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You write brief, plain-language, informational explanations of laboratory results "
    "for a layperson. You are NOT a doctor.\n\n"
    "Hard rules:\n"
    "- Do NOT diagnose, and do NOT tell the reader they have or might have any condition.\n"
    "- Do NOT give emergency, treatment, medication, or dosage instructions.\n"
    "- Do NOT state anything with medical certainty; keep it informational, not advice.\n"
    "- Base every statement ONLY on the structured results provided. Do not infer, "
    "convert, or invent values, units, or ranges.\n"
    "- The 'abnormal_flag' field is authoritative: it was computed by the system, not by "
    "you. Do not re-judge whether a value is in range.\n\n"
    "For a result flagged 'low' or 'high', you may note in plain language that it is "
    "outside the typical reference range and suggest discussing it with a healthcare "
    "professional. For an insight, cite the relevant test name(s) in related_tests. If "
    "nothing is noteworthy, return an empty insights list."
)

INSTRUCTION_PREFIX = (
    "Here are the extracted, already-validated results as JSON. Write informational "
    "insights based only on these:\n\n"
)


def generate_insights(ctx: StageContext) -> None:
    """Stage entrypoint: interpret the extracted data into informational insights."""
    extraction = _load_extraction(ctx)
    results = extraction.get("results", [])

    if not results:
        # Nothing to interpret (e.g. a discharge summary with no lab values). Persist an
        # empty, disclaimered payload and skip the paid model call.
        _persist_insights(ctx, {"insights": [], "summary": None, "disclaimer": DISCLAIMER})
        _log(ctx, outcome="succeeded", duration_ms=0)
        return

    instruction = INSTRUCTION_PREFIX + _context_json(extraction)

    started = time.perf_counter()
    try:
        response = ctx.ai.generate_structured(
            system=SYSTEM_PROMPT,
            instruction=instruction,
            json_schema=INSIGHTS_JSON_SCHEMA,
            max_tokens=INSIGHTS_MAX_TOKENS,
            model=ctx.settings.ai_model_insights or None,
        )
    except AIProviderError as exc:
        _log(
            ctx,
            outcome="error",
            error_code="ai_provider_error",
            detail=str(exc),
            duration_ms=elapsed_ms(started),
        )
        raise TransientStageError(f"insights provider error: {exc}") from exc

    duration_ms = elapsed_ms(started)

    if response.refused:
        _log(
            ctx,
            outcome="refused",
            error_code="model_refusal",
            response=response,
            duration_ms=duration_ms,
        )
        raise TransientStageError("insights refused by safety classifier")

    try:
        result = DocumentInsights.model_validate_json(response.text)
    except ValidationError as exc:
        # Never repair invalid model output — record the failure and let it retry.
        _log(
            ctx,
            outcome="validation_failed",
            error_code="invalid_model_output",
            detail=sanitize_validation_error(exc),
            response=response,
            duration_ms=duration_ms,
        )
        raise TransientStageError("insights output failed validation") from exc

    payload = {
        "insights": [i.model_dump() for i in result.insights],
        "summary": result.summary,
        "disclaimer": DISCLAIMER,
    }
    _persist_insights(ctx, payload)
    _log(ctx, outcome="succeeded", response=response, duration_ms=duration_ms)


# --- helpers ----------------------------------------------------------------


def _load_extraction(ctx: StageContext) -> dict[str, Any]:
    data = ctx.session.execute(
        select(AiReportExtraction.data).where(AiReportExtraction.run_item_id == ctx.item_id)
    ).scalar_one_or_none()
    if data is None:
        # Extraction always runs before insights; a missing row means the pipeline was
        # interrupted. Retry restarts from the top and re-extracts.
        raise TransientStageError("extraction result missing for insights")
    return dict(data)


def _context_json(extraction: dict[str, Any]) -> str:
    """The subset of the extraction the model may reason over — including OUR abnormal
    flag, so it uses the deterministic verdict rather than re-judging ranges."""
    rows = [
        {k: r.get(k) for k in ("test_name", "value", "unit", "reference_range", "abnormal_flag")}
        for r in extraction.get("results", [])
    ]
    return json.dumps({"results": rows, "report_date": extraction.get("report_date")})


def _persist_insights(ctx: StageContext, payload: dict[str, Any]) -> None:
    stmt = (
        pg_insert(AiReportInsight)
        .values(
            run_item_id=ctx.item_id,
            document_id=ctx.document_id,
            data=payload,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
        )
        .on_conflict_do_update(
            index_elements=[AiReportInsight.run_item_id],
            set_={
                "document_id": ctx.document_id,
                "data": payload,
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
            },
        )
    )
    ctx.session.execute(stmt)
    ctx.session.commit()


def _log(ctx: StageContext, **kwargs: Any) -> None:
    log_process(
        ctx,
        stage=STAGE_NAME,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        **kwargs,
    )
