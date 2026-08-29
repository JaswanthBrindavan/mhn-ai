"""Insight generation — the third pipeline stage (runs after extraction).

Flow: read this item's validated extraction, ask the model for brief informational
insights over that structured data (never the raw file, so it cannot introduce values
that bypassed extraction), validate with Pydantic (never repaired), attach a fixed
disclaimer, then persist to ``ai_report_insights`` and a process log.

**Insights are clinically directive, by product decision (2026-08-03.)** They name the
risk a pattern carries and recommend concrete action — diet, lifestyle, follow-up tests,
target values — matching the app's existing Risk Patterns and Suggestions screens. They
were informational-only until that date; see ``Insight`` for what changed and why.

The one line the prompt still holds: **no medication, no dosage, no starting or stopping
a drug, and no emergency instruction.** Recommending a blood test or a dietary change is
not prescribing; recommending a medicine is. A fixed disclaimer is stored with every
payload regardless.

When there is nothing to interpret — no extracted results, or every result determined to
be in range — the model call is skipped entirely.

Idempotent: the insights row and the process log are upserted, so a redelivery that
re-runs the stage overwrites its own prior attempt rather than duplicating rows.
"""

import json
import logging
import time
from functools import partial
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.integrations.ai.base import AIProviderError
from app.models.ai_results import AiReportExtraction, AiReportInsight
from app.services.ai_logging import (
    check_response,
    elapsed_ms,
    log_process,
    sanitize_validation_error,
)
from app.workers.stagetypes import StageContext, TransientStageError

logger = logging.getLogger(__name__)

PROMPT_VERSION = "ins-2026-08-29"
#: ins-5 shortened ``risk_patterns`` (60 -> 40 words) and ``summary`` (unbudgeted -> 60),
#: with their caps moved down to match. The shape is unchanged — same five fields, same
#: rules — but payloads either side of the boundary are not comparable in LENGTH, which
#: is the whole point of the change. Same reasoning ``extraction`` records for ext-3,
#: which was also a cap move and nothing else.
SCHEMA_VERSION = "ins-5"
STAGE_NAME = "generating_insights"
#: Headroom, not a target: the fields are individually capped and a typical report now
#: lands well under this. Truncation IS detected — ``check_response`` ends the item
#: ``failed`` with ``response_truncated`` after one attempt rather than burning the cap
#: (PR #26); this comment used to claim otherwise. Kept generous anyway, because a caught
#: truncation is still a report whose insights never reached the reader.
INSIGHTS_MAX_TOKENS = 16384

#: Stored with every insights payload. Informational framing is not left to the model.
DISCLAIMER = (
    "These insights are informational only and are not a medical diagnosis or advice. "
    "Discuss your results with a qualified healthcare professional."
)

#: Stored instead of insights when every result was determined to be in range. States
#: what the data shows, without interpreting it.
ALL_IN_RANGE_SUMMARY = "All extracted results fall within their reference ranges."


class Insight(BaseModel):
    """One finding, shaped for the app's two cards.

    ``heading`` + ``risk_patterns`` render the Risk Patterns card; ``suggestion_heading``
    + ``suggestions`` render the Suggestions card; ``explanation`` is the plain-language
    line behind them.

    Every field is capped short on purpose. This is read on a phone by someone with no
    medical training, and a long block does not get read at all — so the caps are a
    product requirement, not storage hygiene. ``explanation`` merges what were two
    separate fields that each explained the same thing at length.

    **The prompt states a word budget for each field, and these caps sit above it.** A cap
    the model is not told about is not a limit, it is a paid failure: over-long output
    fails validation, which this stage treats as transient, so it retries the whole call
    at full price. A 400-char ``risk_patterns`` cap with no stated budget did exactly that
    once, at $0.046 per attempt. Tighten a cap and the budget together, or not at all —
    which is how ``risk_patterns`` and ``summary`` were shortened on 2026-08-29.

    One exception, deliberately left: ``heading`` has a cap and no budget. It is a card
    title of a few words in practice, so 200 characters is not a limit it can reach — but
    it is the one field where the paragraph above is aspiration rather than fact.

    **These are clinically directive, by product decision (2026-08-03).** They name
    conditions, state the risk a pattern carries, and recommend concrete actions — diet,
    lifestyle, follow-up tests, target values. That is a deliberate change from the
    earlier informational-only framing, made after comparing both against the app's
    existing Risk Patterns / Suggestions screens.

    The line that remains: **no medication, no dosage, no starting or stopping a drug,
    and no emergency instruction.** Recommending a blood test or a dietary change is not
    prescribing; recommending a medicine is. The disclaimer is still attached to every
    payload.
    """

    #: Names the finding AND its risk, as the Risk Patterns card title:
    #: "Elevated Uric Acid - Gout & Renal Risk".
    heading: str = Field(min_length=1, max_length=200)
    #: One or two lines: what this test looks at and what moves it, together. Merged
    #: from two fields that were separately explaining the same thing at length.
    explanation: str = Field(min_length=1, max_length=350)
    #: The Risk Patterns card body: the value against the limit it crossed, then what
    #: that can lead to. Two short lines.
    #:
    #: 350, down from 500, with the prompt's budget cut 60 -> 40 words in the same
    #: change. 40 is close to the floor rather than an arbitrary trim: the field carries
    #: three things — the value, the limit from ``flagged_against``, and the consequence
    #: — and the prompt's own example spends 24 words on all three. Below about 25 the
    #: limit citation is what a model drops, and that citation is what stops it quoting a
    #: number the value never crossed.
    risk_patterns: str = Field(min_length=1, max_length=350)
    #: The Suggestions card title — an action, e.g. "Reduce Uric Acid Through Diet".
    suggestion_heading: str = Field(min_length=1, max_length=120)
    #: The Suggestions card body: concrete steps, two or three short lines. Never
    #: medication, dosage, or starting/stopping a drug.
    suggestions: str = Field(min_length=1, max_length=500)
    #: Test names from the extraction this insight refers to.
    related_tests: list[str] = Field(default_factory=list)


class DocumentInsights(BaseModel):
    """Validated model output. An empty list is valid (nothing noteworthy to say)."""

    insights: list[Insight]
    #: 700, down from 2000, against a stated budget of 60 words. It had NO budget at all
    #: before — the exact trap ``Insight`` describes, sitting on the longest field here.
    #: The ratio of cap to budget is looser than ``risk_patterns``' on purpose: this is
    #: also where unchecked tests are named in a clause, and a list of test names spends
    #: characters without spending words.
    summary: str | None = Field(default=None, max_length=700)


_INSIGHT_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "heading": {"type": "string"},
        "explanation": {"type": "string"},
        "risk_patterns": {"type": "string"},
        "suggestion_heading": {"type": "string"},
        "suggestions": {"type": "string"},
        "related_tests": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "heading",
        "explanation",
        "risk_patterns",
        "suggestion_heading",
        "suggestions",
        "related_tests",
    ],
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
    "You write a lab report's Risk Patterns and Suggestions for the person whose report "
    "it is. Be concrete and clinically useful: name the pattern, say what risk it "
    "carries, and give steps worth acting on.\n\n"
    "Hard rules:\n"
    "- NEVER name, recommend, adjust or discourage a MEDICATION, supplement dose, or "
    "any drug — prescription or over the counter. Recommending a blood test or a change "
    "of diet is fine; recommending a medicine is not, ever.\n"
    "- NEVER give emergency instructions or tell the reader to seek urgent care.\n"
    "- Base every statement ONLY on the structured results provided. Do not infer, "
    "convert, or invent values, units, or ranges. Every number you cite must appear in "
    "the input, and any limit you quote must come from that result's 'flagged_against'.\n"
    "- The 'abnormal_flag' field is authoritative: it was computed by the system, not by "
    "you. Do not re-judge whether a value is in range.\n"
    "- Do NOT add a 'discuss this with your doctor/healthcare provider' line to your "
    "insights. A disclaimer saying exactly that is attached to every set of insights; "
    "repeating it per result is noise.\n\n"
    "Write ONE insight per finding, not one per row, and MERGE AGGRESSIVELY. Results "
    "belong in the same insight when they describe one finding: a percentage and its "
    "absolute count, a ratio and the values it is derived from, or several markers of "
    "the same body system telling the same story (the red-cell markers of anaemia; "
    "sodium and chloride together; the liver enzymes). List every test name you covered "
    "in related_tests. Every flagged result must be COVERED by some insight: merge, "
    "never drop.\n\n"
    "WRITE IN SIMPLE ENGLISH. This is read on a phone by someone with no medical "
    "training. A long block does not get read at all, so short beats complete:\n"
    "- Almost no medical words. Where an everyday word exists, use it: 'bile' not "
    "'cholestatic', 'liver cells' not 'hepatocellular', 'paler than usual' not "
    "'hypochromic', 'joint pain' not 'gout flares', 'kidney' not 'renal'. If a term has "
    "no everyday equivalent, put the plain meaning in brackets right after it once.\n"
    "- Short sentences, one idea each. No semicolons stacking three clauses together.\n"
    "- Keep the numbers — they are the point — but drop unit strings the reader cannot "
    "use if the sentence already reads clearly.\n"
    "- Do not repeat between fields. Say a thing once.\n\n"
    "Each insight fills five fields. STAY INSIDE THE LINE LIMITS:\n\n"
    "- heading: the Risk Patterns card title. Name the finding AND what it can lead to, "
    "joined by a dash, in plain words. 'High Uric Acid - Joint Pain & Kidney Risk'. "
    "'Slightly High LDL Cholesterol - Heart Risk'. Where several results form one "
    "pattern, name it once: 'Low Iron-Related Blood Markers - Possible Nutritional Gap'.\n"
    "- explanation: ONE OR TWO LINES, AT MOST 40 WORDS, covering both what this "
    "test looks at and what commonly moves it. 'Uric acid is a waste "
    "product your kidneys clear out. It builds "
    "up when you eat a lot of red meat or shellfish, drink alcohol, or do not drink "
    "enough water.'\n"
    "- risk_patterns: TWO SHORT LINES, AT MOST 40 WORDS. Start with the "
    "value and the limit it crossed. The limit MUST be taken from 'flagged_against', "
    "which is the range the result was actually checked against — never a number from "
    "anywhere else. Then say "
    "plainly what that can lead to. 'Uric acid "
    "is 8.6, above the normal top of 8.0. Staying this high can cause sudden joint pain "
    "and, over time, kidney stones.' Say when something is only mild — 'this is only "
    "slightly above the line' — so the reader can tell small from serious.\n"
    "- suggestion_heading: the Suggestions card title. An action, AT MOST 6 WORDS: "
    "'Cut Down Uric Acid Through Food'. 'Check for Low Iron'.\n"
    "- suggestions: TWO OR THREE SHORT LINES, AT MOST 60 WORDS, of things to "
    "actually do. Name real foods, name the follow-up test, give the "
    "retest gap. 'Eat less red meat, organ meat and "
    "shellfish, and cut back on alcohol. Drink more water. Get uric acid checked again "
    "in 4 to 6 weeks.' Say what to do, not who to ask.\n"
    "  BANNED here, because they are true of every result and so say nothing: 'discuss "
    "this with your doctor', 'ask a clinician', 'consult a healthcare professional', "
    "'interpret alongside your other results', 'a clinician can advise'. A disclaimer "
    "on every payload already says that. Give the reader something to act on instead.\n\n"
    "**Write an insight ONLY for results flagged 'low' or 'high'.** Nothing else earns "
    "one:\n"
    "- A result flagged 'normal' NEVER gets its own insight. Use it as context inside "
    "another finding if it sharpens the picture, and nothing more.\n"
    "- **Never write a round-up insight** — no 'Remaining Tests - All Within Normal "
    "Limits', no 'Other Results', no 'Everything Else Is Fine'. The summary already "
    "covers what came back normal. A card that says nothing happened is a card the "
    "reader has to swipe past to reach one that matters.\n"
    "- A result flagged null means the system could not check it against any range. Do "
    "NOT give it its own insight and do NOT judge whether it is in range. Name those "
    "tests in the SUMMARY instead, in one clause — 'X and Y were reported without "
    "reference ranges, so they could not be checked'.\n\n"
    "summary: TWO OR THREE SHORT SENTENCES, AT MOST 60 WORDS, covering the whole panel — "
    "what was flagged, grouped sensibly, and what came back within range. Written for "
    "someone reading it before any of the detail below it."
)

INSTRUCTION_PREFIX = (
    "Here are the extracted, already-validated results as JSON. Write informational "
    "insights based only on these:\n\n"
)


def generate_insights(ctx: StageContext) -> None:
    """Stage entrypoint: interpret the extracted data into informational insights."""
    extraction = _load_extraction(ctx)
    results = extraction.get("results", [])

    if not _needs_interpretation(results):
        # Nothing to interpret: no lab values at all (e.g. a discharge summary), or every
        # value determined in range. Persist a disclaimered payload, skip the paid call.
        summary = ALL_IN_RANGE_SUMMARY if results else None
        _persist_insights(ctx, {"insights": [], "summary": summary, "disclaimer": DISCLAIMER})
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

    # Refusal (transient) and truncation (permanent) are the same check for every
    # stage, so it lives in one place; partial binds this stage's own log helper.
    check_response(
        response,
        log=partial(_log, ctx),
        duration_ms=duration_ms,
        what="insights",
    )

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


def _needs_interpretation(results: list[dict[str, Any]]) -> bool:
    """Whether the model is worth paying for: is any result not known to be in range?

    ``abnormal_flag`` is 'low' | 'normal' | 'high' | None, where None means the value or
    its range could not be parsed. Undetermined is NOT normal — those still go to the
    model, so a report we could not check never gets described as all-clear.
    """
    return any(r.get("abnormal_flag") != "normal" for r in results)


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
    flag, so it uses the deterministic verdict rather than re-judging ranges.

    ``flagged_against`` is here and ``reference_range`` is NOT, and that is the point. The
    two are the same number until an approved ideal range is in play, and then they are
    not: the flag is computed against R&D's age-bracket bounds while the report goes on
    printing the lab's own. A model shown only the printed range would write "8.6, above
    the normal top of 8.0" about a value that crossed a different limit entirely — citing
    a number the reader can see on their own report, beside a verdict it does not support.

    Sending one range rather than both is deliberate. The prompt tells the model to quote
    the limit that was crossed; given two, it has to choose, and that is a judgement this
    stage exists to keep away from the model.
    """
    rows = [
        {k: r.get(k) for k in ("test_name", "value", "unit", "flagged_against", "abnormal_flag")}
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
