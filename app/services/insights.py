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
from datetime import date
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
from app.services.dates import parse_date
from app.services.extraction import name_key
from app.workers.stagetypes import StageContext, TransientStageError

logger = logging.getLogger(__name__)

#: ins-2026-08-31 rewrote the `suggestions` guidance. It listed "name real foods, name
#: the follow-up test, give the retest gap" as three equal things, and a model can satisfy
#: that with the last two — so most suggestions came back as nothing but "get retested in
#: N weeks", which is nothing to do for N weeks. Lifestyle first is now the rule, a test is
#: the last line, and inventing a diet for a marker diet does not move is banned outright.
#: ins-2026-08-31b states each field's CHARACTER limit beside its word budget. Words were
#: the only unit given before, and a model counts them badly enough that `summary` came
#: back at 779 against a 700-char cap it had never been told about.
#: ins-2026-09-01 stopped sending results flagged `normal`. Every flag is computed in
#: Python — against the R&D ideal range where one resolved — so a normal result is a
#: settled question, and the prompt already forbade giving one its own insight.
#:
#: **Measured both ways on two real full-body panels before changing anything**
#: (`docs/live-runs/2026-09-01-insights-payload-ab.md`), because the objection to it was
#: that an insight might quietly lean on a normal result for context:
#:
#: * **it does not.** Across 48- and 73-result panels and ten insights, **not one normal
#:   result was cited in `related_tests`**. The context those rows were paying for was
#:   never used.
#: * **coverage did not regress; it improved.** On the 73-result panel the full payload
#:   missed two flagged results, the trimmed one missed one.
#: * **the summary got more accurate, which was not the point but is the best part.**
#:   Asked to count normals from rows, the model got it wrong both times (28 against 32,
#:   45 against 62). Given `normal_count` it was exact both times. Counting rows is
#:   arithmetic, and this codebase does arithmetic in Python.
#: * cost: input tokens down 43% and 61%, but **output did not shrink** (it grew 22% on one),
#:   and output bills at 5x input — so the real saving is ~19% of the stage on Sonnet
#:   prices, not the 60% the input cut suggests. Worth having, smaller than it looks.
#:
#: **The residual risk, stated plainly:** an insight can no longer use a normal result as
#: context — a high neutrophil percentage beside a normal absolute count now reaches the
#: model as the percentage alone. The measurement says that is not happening today; it does
#: not prove it never would. And both arms ran on Gemini because the Anthropic account was
#: out of credit, so Sonnet's PROSE under this payload is still unverified.
PROMPT_VERSION = "ins-2026-09-01"
#: ins-5 shortened ``risk_patterns`` (60 -> 40 words) and ``summary`` (unbudgeted -> 60),
#: with their caps moved down to match. The shape is unchanged — same five fields, same
#: rules — but payloads either side of the boundary are not comparable in LENGTH, which
#: is the whole point of the change. Same reasoning ``extraction`` records for ext-3,
#: which was also a cap move and nothing else.
#: ins-6 raised the ``risk_patterns`` and ``explanation`` caps back to 500 after ins-5's
#: 350 started rejecting real output. The prompt's word budgets are untouched, so payloads
#: either side of this boundary ARE comparable — unlike ins-5's, which is why that one is
#: called out above and this one is recorded rather than explained.
#: ins-7 split the length the model is ASKED for (``FIELD_LIMITS``, stated in the prompt)
#: from the length the validator REJECTS at (twice that, a runaway guard). Payloads either
#: side are comparable in shape; lengths are not, because everything up to ins-6 asked for
#: a length in words and enforced a different one, in characters, in code.
SCHEMA_VERSION = "ins-7"
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


#: How long each field should be, in characters. Used twice: the prompt, which STATES it,
#: and the validator, which sits well above it.
#:
#: **A length cannot be enforced on the model, only asked for.** Anthropic's structured
#: outputs do not support `minLength`/`maxLength` — they are on the documented unsupported
#: list, alongside numeric bounds and recursive schemas, and the SDK strips unsupported
#: keywords out of the schema before sending it. So putting them in
#: `_INSIGHT_ITEM_SCHEMA` would not constrain generation; it would read as a guarantee and
#: be silently discarded, which is worse than not being there. That was tried on
#: 2026-08-31 and reverted before it shipped.
#:
#: What IS available is the prompt, and there the unit matters: the fields carried only a
#: WORD budget until now, and `summary` came back at 779 characters against a 700-character
#: cap it had never been told about in any unit. Every field now states both.
#:
#: The consequence is that these numbers are a request and the validator must be forgiving
#: — see `_RUNAWAY_FACTOR`. Do not tighten the cap towards this value on the theory that
#: the model will comply; nothing makes it comply.
FIELD_LIMITS: dict[str, int] = {
    "heading": 200,
    "explanation": 500,
    "risk_patterns": 500,
    "suggestion_heading": 120,
    "suggestions": 700,
    "summary": 700,
}

#: How far above the stated limit the VALIDATOR sits.
#:
#: The two numbers do different jobs and conflating them is what kept breaking. The limit
#: above is a product decision — how much a person will read on a phone — and it belongs in
#: the schema and the prompt, where it can actually shorten the output. This one is a
#: runaway guard: it exists to reject a model that returns a page of prose, not to police
#: the last 11% of a paragraph.
#:
#: A validation failure here is TRANSIENT, so a field over the line costs three paid
#: retries and then the reader gets no insights at all. Set the guard where that is worth
#: it — an answer twice as long as asked is broken; one slightly over is just long.
_RUNAWAY_FACTOR = 2


def _cap(field: str) -> int:
    """The validator's ceiling for a field: its stated limit, doubled. See above."""
    return FIELD_LIMITS[field] * _RUNAWAY_FACTOR


class Insight(BaseModel):
    """One finding, shaped for the app's two cards.

    ``heading`` + ``risk_patterns`` render the Risk Patterns card; ``suggestion_heading``
    + ``suggestions`` render the Suggestions card; ``explanation`` is the plain-language
    line behind them.

    Every field is capped short on purpose. This is read on a phone by someone with no
    medical training, and a long block does not get read at all — so the caps are a
    product requirement, not storage hygiene. ``explanation`` merges what were two
    separate fields that each explained the same thing at length.

    **The length the model is ASKED for is ``FIELD_LIMITS``; the number here is a runaway
    guard at twice that.** They are different jobs and conflating them cost three payloads
    in one day. A length cannot be enforced on the model — structured outputs do not
    support ``maxLength`` — so ``FIELD_LIMITS`` is a request stated in the prompt, and
    this is the point at which an answer is thrown away instead of merely read long.

    That distinction is the whole lesson. A validation failure here is TRANSIENT: three
    paid retries and then the reader gets no insights at all. Worth it for a model
    returning a page of prose; not worth it for a paragraph 11% over, which is exactly
    what ``summary`` did at 779 against 700.

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
    heading: str = Field(min_length=1, max_length=_cap("heading"))
    #: One or two lines: what this test looks at and what moves it, together. Merged
    #: from two fields that were separately explaining the same thing at length.
    #:
    #: 500, raised from 350 alongside ``risk_patterns`` and for the same reason — it is
    #: the other 40-word field and carried the identical cap, so it fails the identical
    #: way. Fixed together rather than one at a time.
    explanation: str = Field(min_length=1, max_length=_cap("explanation"))
    #: The Risk Patterns card body: the value against the limit it crossed, then what
    #: that can lead to. Two short lines.
    #:
    #: The 40-word budget in the prompt is unchanged and is the product control. This cap
    #: is a safety net, and **the net firing is worse than the thing it catches**: the
    #: stage treats a validation failure as transient, so an over-long field costs three
    #: paid retries and then the reader gets NO insights at all — strictly worse than one
    #: paragraph running long.
    #:
    #: 500, back up from the 350 set on 2026-08-29. That change cut the budget 60 -> 40
    #: words and moved the cap down "to match", which left only ~8.7 characters per
    #: budgeted word — and this is the densest field there is, naming a value, a unit, a
    #: limit and a consequence, and grouping several results when they form one pattern.
    #: It fired in production on 2026-08-31 (`insights.0.risk_patterns: string_too_long`),
    #: on the first reports processed after the approved-THP override went on: a tighter
    #: range flags more results, more results group into one pattern, and the pattern is
    #: written into this field.
    #:
    #: The rule the class docstring states still holds — never tighten a cap without
    #: tightening the budget with it. What is added here is its other half: never size a
    #: cap so close to the budget that an ordinary overshoot destroys the payload.
    risk_patterns: str = Field(min_length=1, max_length=_cap("risk_patterns"))
    #: The Suggestions card title — an action, e.g. "Reduce Uric Acid Through Diet".
    suggestion_heading: str = Field(min_length=1, max_length=_cap("suggestion_heading"))
    #: The Suggestions card body: concrete steps, two or three short lines. Never
    #: medication, dosage, or starting/stopping a drug.
    #:
    #: 700, raised with the ins-2026-08-31 prompt and because of it. This field now has to
    #: carry a lifestyle change AND the follow-up, where before a model could satisfy the
    #: instruction with the follow-up alone — so the same 60-word budget is being asked to
    #: hold more, against what was the tightest cap-to-budget ratio left (8.3 characters
    #: per budgeted word). Raising it in the same change is the point: the failure this
    #: whole set of comments exists for is a cap that fires because the budget moved
    #: underneath it.
    suggestions: str = Field(min_length=1, max_length=_cap("suggestions"))
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
    summary: str | None = Field(default=None, max_length=_cap("summary"))


_INSIGHT_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # No `maxLength` here, and it is not an oversight — see FIELD_LIMITS.
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
    "- heading: the Risk Patterns card title, AT MOST 200 CHARACTERS. Name the finding "
    "AND what it can lead to, "
    "joined by a dash, in plain words. 'High Uric Acid - Joint Pain & Kidney Risk'. "
    "'Slightly High LDL Cholesterol - Heart Risk'. Where several results form one "
    "pattern, name it once: 'Low Iron-Related Blood Markers - Possible Nutritional Gap'.\n"
    "- explanation: ONE OR TWO LINES, AT MOST 40 WORDS / 500 CHARACTERS, covering both what this "
    "test looks at and what commonly moves it. 'Uric acid is a waste "
    "product your kidneys clear out. It builds "
    "up when you eat a lot of red meat or shellfish, drink alcohol, or do not drink "
    "enough water.'\n"
    "- risk_patterns: TWO SHORT LINES, AT MOST 40 WORDS / 500 CHARACTERS. Start with the "
    "value and the limit it crossed. The limit MUST be taken from 'flagged_against', "
    "which is the range the result was actually checked against — never a number from "
    "anywhere else. Then say "
    "plainly what that can lead to. 'Uric acid "
    "is 8.6, above the normal top of 8.0. Staying this high can cause sudden joint pain "
    "and, over time, kidney stones.' Say when something is only mild — 'this is only "
    "slightly above the line' — so the reader can tell small from serious.\n"
    "- suggestion_heading: the Suggestions card title. An action, AT MOST 6 WORDS / "
    "120 CHARACTERS: "
    "'Cut Down Uric Acid Through Food'. 'Check for Low Iron'.\n"
    "- suggestions: TWO OR THREE SHORT LINES, AT MOST 60 WORDS / 700 CHARACTERS, of things to "
    "actually do. Say what to do, not who to ask.\n"
    "  START with what the reader can change THEMSELVES — food, drink, movement, sleep, "
    "sunlight, weight, alcohol, tobacco — named specifically. 'Eat better' and 'improve "
    "your diet' are not suggestions; 'eat more spinach, rajma and dates' is. Name foods "
    "and habits an ordinary Indian household already recognises.\n"
    "  A FOLLOW-UP TEST IS THE LAST LINE, NEVER THE WHOLE ANSWER. A reader told only to "
    "get retested in six weeks has been given nothing to do for six weeks, which is "
    "exactly the stretch where a change would show. Give the retest gap once the "
    "actions are there.\n"
    "  Where the marker genuinely has no lifestyle lever — most red-cell indices, "
    "platelet size and distribution, a ratio derived from other results — say so plainly "
    "in a few words and give the follow-up instead. **Do not invent a diet for a number "
    "that diet does not move**: a made-up food fix is worse than admitting there is "
    "none, because the reader will follow it and believe they are treating something.\n"
    "  'Eat less red meat, organ meat and shellfish, and cut back on alcohol. Drink more "
    "water. Get uric acid checked again in 4 to 6 weeks.'\n"
    "  BANNED here, because they are true of every result and so say nothing: 'discuss "
    "this with your doctor', 'ask a clinician', 'consult a healthcare professional', "
    "'interpret alongside your other results', 'a clinician can advise'. A disclaimer "
    "on every payload already says that. Give the reader something to act on instead.\n\n"
    "**Write an insight ONLY for results flagged 'low' or 'high'.** Nothing else earns "
    "one:\n"
    "- **Results that came back NORMAL are not given to you.** Only the ones that need "
    "interpreting are: everything flagged 'low' or 'high', plus anything the system could "
    "not check. How many were normal is in 'normal_count'. Do not ask for the others, do "
    "not guess which tests they were, and never imply you have seen a result you were not "
    "given.\n"
    "- **Never write a round-up insight** — no 'Remaining Tests - All Within Normal "
    "Limits', no 'Other Results', no 'Everything Else Is Fine'. The summary already "
    "covers what came back normal. A card that says nothing happened is a card the "
    "reader has to swipe past to reach one that matters.\n"
    "- A result flagged null means the system could not check it against any range. Do "
    "NOT give it its own insight and do NOT judge whether it is in range. Name those "
    "tests in the SUMMARY instead, in one clause — 'X and Y were reported without "
    "reference ranges, so they could not be checked'.\n\n"
    "summary: TWO OR THREE SHORT SENTENCES, AT MOST 60 WORDS / 700 CHARACTERS, covering "
    "the whole panel — what was flagged, grouped sensibly, and how many came back within "
    "range. Use 'normal_count' for that last part and state it as a number ('the other 41 "
    "results were within their normal ranges'); you were not given those results, so do "
    "not name them. Written for someone reading it before any of the detail below it."
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


#: The fields of a result the model may see. ``flagged_against`` is here and
#: ``reference_range`` is NOT, and that is the point. The two are the same number until an
#: approved ideal range is in play, and then they are not: the flag is computed against
#: R&D's age-bracket bounds while the report goes on printing the lab's own. A model shown
#: only the printed range would write "8.6, above the normal top of 8.0" about a value that
#: crossed a different limit entirely — citing a number the reader can see on their own
#: report, beside a verdict it does not support. Sending one range rather than both is
#: deliberate: the prompt tells the model to quote the limit that was crossed, and given two
#: it would have to choose, which is a judgement this stage exists to keep away from it.
_SENT_FIELDS = ("test_name", "value", "unit", "flagged_against", "abnormal_flag")


def _current_only(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop a result that a later observation of the SAME test supersedes.

    A cumulative report prints the same analyte across several visits — "YOUR CURRENT
    VISIT" beside "FROM YOUR PREVIOUS 3 VISITS" — and ``extraction._dedupe_results`` keeps
    every one of them **on purpose**: the trend is what such a report is for, and its
    docstring says losing half of it is silent. That is right for what is STORED.

    It is wrong for what is interpreted. ``_context_json`` sends no ``observed_date``, so
    the model cannot tell a current value from a sixteen-month-old one and will write
    about whichever it is handed. On the report that found this (document 32): current
    triglycerides 139, in range — and the 2024 column's 219, out of range. The reader
    would have been told their triglycerides are high, with a real number, from a value
    that is no longer true.

    **The normals filter above made that certain rather than likely.** The current 139 is
    ``normal`` and is dropped there, so the stale 219 was the only triglycerides row left
    in the payload. Two changes that are each correct alone, and together produce a
    confident falsehood.

    Only a STRICTLY LATER parseable date supersedes, which is the refuse-rather-than-guess
    rule the rest of this pipeline follows. Two undated rows both survive; a dated row
    never displaces an undated one, because "no date" is not evidence of being older. So
    an ordinary single-visit report is untouched — nothing in it has a later twin.
    """
    latest: dict[str, date] = {}
    for r in results:
        when = parse_date(r.get("observed_date"))
        if when is None:
            continue
        key = _test_key(r)
        if key not in latest or when > latest[key]:
            latest[key] = when

    kept = []
    for r in results:
        when = parse_date(r.get("observed_date"))
        newest = latest.get(_test_key(r))
        if when is not None and newest is not None and when < newest:
            continue
        kept.append(r)
    return kept


def _test_key(result: dict[str, Any]) -> str:
    """Same-test identity, matching ``extraction._dedupe_results``' own key."""
    return name_key(str(result.get("test_name") or ""))


def _context_json(extraction: dict[str, Any]) -> str:
    """What the model may reason over: the results that need interpreting, and nothing else.

    **Results flagged ``normal`` are not sent** (2026-09-01). Every flag is already computed
    in Python — against the R&D-approved ideal range where one resolved, the report's own
    printed range otherwise — so a normal result is a settled question, and the prompt has
    always forbidden giving one its own insight. Sending forty of them bought one sentence
    in the summary and paid for the whole panel to be re-read. They are still extracted,
    still stored, and still shown to the reader in the results table; they are simply not
    part of what the model is asked about.

    **Results with a NULL flag are sent, and that is not an oversight.** Null means the
    system could not check the value against any range at all — no approved THP matched, or
    the report printed a unit with no curated conversion — which is a different thing from
    "in range" and the one case the reader most needs named. The prompt requires them
    listed in the summary, so they have to be in the payload to be listed.

    **``normal_count`` replaces the rows it stands for.** The summary is required to say
    what came back within range, and with the normals gone the model would otherwise have
    to either omit that or invent it. A count is the whole of what that sentence needs, and
    it cannot be misquoted as a value.
    """
    results = _current_only(extraction.get("results", []))
    rows = [
        {k: r.get(k) for k in _SENT_FIELDS} for r in results if r.get("abnormal_flag") != "normal"
    ]
    return json.dumps(
        {
            "results": rows,
            "normal_count": sum(1 for r in results if r.get("abnormal_flag") == "normal"),
            "report_date": extraction.get("report_date"),
        }
    )


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
