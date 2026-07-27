# Section extraction (insurance · scans/imaging · vaccinations)

The report pipeline is `classify -> extract -> insights`, and it deep-processes only
documents the classifier places in `reports`. Every other section it recognises is
correctly identified and then rejected, with the section name as the reject reason:

```python
# app/services/classification.py
PROCESSABLE_SECTIONS: frozenset[DocumentSection] = frozenset({DocumentSection.REPORTS})
```

This package adds one stage, `extract_section`, that handles three of those sections by
reading the document's **text** and transcribing its fields into `ai_section_extractions`.

## What it does

| Section | Fields extracted |
|---|---|
| `insurance` | insurer · policy name/type · co-pay · policy period · ≤5 covered conditions · ≤5 exclusions |
| `scans_imaging` | scan type · body part · scan date · facility · plain-language summary · impression · ≤5 findings |
| `vaccinations` | title · vaccine · dose · date given · next dose due · facility |

It follows the same rules as the existing stages: structured output under a fixed JSON
schema, validated with Pydantic and **never repaired**, upserted so a redelivery
overwrites its own attempt, and every model call logged to `ai_process_logs` by
`(run_item_id, stage, attempt)`.

## OCR-first, not vision

The report pipeline hands the raw PDF to Claude and lets it read the file. This package
does not: it extracts text itself, then sends only that text to the model.

```
PDF/image ──► text layer present?  ──yes──► PyMuPDF reads it        (exact, ~5-180ms)
                    │
                    no
                    ▼
              rasterise at 200 DPI ──► Tesseract OCR   (~1-3s per page, scored)
```

Measured on the sample documents:

| Document | Text layer | Same file scanned | OCR confidence |
|---|---|---|---|
| Chest X-Ray | 1,424 chars, 3 ms | 1,358 chars, 3.0 s | 0.91 |
| Insurance receipt | 1,669 chars, 9 ms | 1,467 chars, 2.0 s | 0.91 |
| Vaccination certificate | 1,251 chars, 48 ms | 1,352 chars, 1.2 s | 0.76 |

All twelve sample PDFs are digital, so in practice the text layer handles them and OCR
never runs. OCR exists for the phone photo of a vaccination card.

What this buys, beyond cost: **a failure becomes attributable.** The OCR engine, page
counts and mean confidence are stored beside every extraction, so a missing field can be
traced to a bad scan rather than blamed on the model. A read below 0.60 confidence is
flagged `low_ocr_confidence`, and pages dropped past the 30-page cap are flagged
`pages_not_read`.

What it costs: **OCR is now the accuracy ceiling.** Anything Tesseract drops, the model
cannot recover, because it never sees the original. That is the opposite of the bet the
report pipeline makes, and it is worth a conscious decision rather than drift.

## Two deliberate choices

**Dates are normalised in Python, not trusted from the model.** The prompt asks for
`DD/MM/YYYY`, but documents print ordinals (`28th July 2026`), DICOM study dates
(`20210831`), and radiology timestamps (`31/08/2021 18:11:28`) — and a prompt is a
request, not a guarantee. `app.insights.dates` parses all of those and stores ISO. An
unreadable date becomes `null` rather than a guess. This mirrors extraction computing
abnormal flags in Python: a deterministic rule beats an instruction.

**Bad date pairs are flagged, not asserted.** A policy whose end date precedes its start
is a misread or a bad document; either way "is this still active?" cannot be answered
from it. The values are stored with a `dates_out_of_order` flag rather than silently
presented as fact.

## Wiring it in

The stage is written and tested but **not connected** — `STAGE_SEQUENCE` is unchanged, so
behaviour today is exactly as before. Connecting it needs two decisions from the team,
which is why it is left out of this change:

**1. Let the classifier through.** Non-report sections are rejected in
`classify_report`. They would need to pass when an extractor exists:

```python
# app/services/classification.py
from app.insights import SUPPORTED_SECTIONS

if result.section not in PROCESSABLE_SECTIONS | SUPPORTED_SECTIONS:
    ...reject as now...
```

**2. Branch the pipeline.** `STAGE_SEQUENCE` is a flat list, so every item runs every
stage. Reports and sections need different paths — a report runs
`extract_report -> generate_insights`, a section runs `extract_section` and stops. That
is a change to the sequence's shape, not a line to append, so it belongs in a design
discussion rather than being smuggled in here.

The `EXTRACTING` status is reused for the section stage; a dedicated status would need a
migration for the `CHECK` constraint on `ai_processing_run_items.status`.

## Adding a section

An entry in `sections.py` — a Pydantic model, a hand-written JSON schema, and a prompt.
Nothing else changes: the stage, the persistence, the logging and the worker are generic.

```python
DocumentSection.BILLS: SectionSpec(
    section=DocumentSection.BILLS,
    model=BillFields,
    json_schema=_BILL_SCHEMA,
    system_prompt=_BILL_PROMPT,
    max_tokens=2048,
    date_fields=("invoice_date",),
)
```

Schemas are hand-written rather than generated from the Pydantic model because
structured output rejects the constraint keywords Pydantic emits — no numeric or length
constraints, `additionalProperties: false`, every property listed in `required`, nullable
expressed as a `["string", "null"]` union. The length limits live on the model, which is
what actually validates.

## Deployment note

OCR needs the **Tesseract binary**, which `pytesseract` only binds to. The Docker image
must install it:

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
```

Without it, digital PDFs still work (the text layer needs no binary) and scanned ones
fail with `text_extraction_failed`. `TESSERACT_CMD` overrides the binary path when it is
not on `PATH`.

**Licensing:** `pymupdf` is AGPL-3.0. Commercial distribution normally requires a paid
Artifex licence — worth confirming with whoever owns licensing before this ships.

## Layout

```
app/insights/
├── ocr.py          text layer (PyMuPDF) -> Tesseract fallback, with confidence
├── dates.py        parse / iso / display, and date-order checking
├── sections.py     one SectionSpec per section: model, schema, prompt
├── extraction.py   the extract_section stage
└── __init__.py

app/models/ai_results.py                     + AiSectionExtraction
alembic/versions/f4b8c2e6a1d9_*.py           creates ai_section_extractions
tests/unit/test_insight_ocr.py               text layer + OCR fallback
tests/unit/test_insight_dates.py             date handling
tests/unit/test_insight_sections.py          specs + stored payload
tests/integration/test_section_extraction.py the stage against DB + moto S3
```
