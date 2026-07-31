<a id="readme-top"></a>

<div align="center">
  <h1>MHN AI</h1>
  <p>
    AI-assisted medical-document processing for MyHealthNotion.
    <br />
    Classifies uploaded documents, extracts structured lab data, and generates
    informational insights — asynchronously, idempotently, and cost-logged.
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul><li><a href="#built-with">Built With</a></li></ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About The Project

Users upload medical documents through the MyHealthNotion app. Spring stores each upload
in the `unclassified_files` table and calls this service with the document ids. From there
this service is both the classifier and the router:

1. **Classify** the document into an app section — `reports`, `scans_imaging`,
   `prescriptions`, `insurance`, `bills`, `vaccinations`, `medical_condition`, or `unknown`.
2. **If it is a report:** extract the lab results, generate insights, then move the document
   into the `reports` table and write the assembled payload to `reports.content` — the
   insert, the content write, and the delete from `unclassified_files` all happen in one
   transaction, so a document is never in both tables or in neither.
3. **Anything else** is recorded with its detected section and left in `unclassified_files`
   for a later sprint to handle.

Design notes worth knowing before reading the code:

- **The API never does AI work inline.** `POST` persists the run, publishes to SQS, and
  returns `202 Accepted`. Workers do the processing and scale independently.
- **SQS is at-least-once, so everything is idempotent.** A partial unique index allows one
  in-flight item per document, stages upsert their results, and the move is atomic — a
  redelivered message can only redo work, never duplicate it.
- **The model transcribes; Python decides.** Abnormal/out-of-range flags and unit conversion
  are deterministic application code, never model arithmetic. Model output is validated with
  Pydantic and never silently repaired. Where a reading is genuinely ambiguous — a censored
  value like `< 200` against a range it might sit either side of — the flag is left unset
  rather than guessed.
- **Insights are informational only** — no diagnosis, no emergency instruction, no medical
  certainty. A disclaimer is always stored alongside them.
- **Source files stay private in S3.** The service handles object keys, never public URLs.

Authentication is service-level: a shared bearer token proves the caller is the Spring
backend. This service performs no user-level authorization — Spring has already made that
decision and re-authorizes on read.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

- Python 3.12
- FastAPI + Pydantic v2
- SQLAlchemy 2.x + Alembic
- PostgreSQL
- AWS S3 (source documents) + SQS (work queue, with a DLQ)
- Anthropic Claude (per-stage models: a fast model for classification and extraction, a
  stronger one for insights)
- pypdfium2 — trims documents for the classifier, and rasterises pages for OCR
- pdfplumber — reads the text layer sorted by position on the page
- Tesseract (via pytesseract) — OCR for scanned pages, needs the binary in the image
- Docker Compose

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

- Python 3.12+ and [uv](https://github.com/astral-sh/uv)
- Docker Desktop (for the containerised API, worker, and local Postgres)
- Access to the PostgreSQL database carrying the Spring base schema
- AWS credentials for the source S3 bucket and the processing queues
- An `ANTHROPIC_API_KEY`

### Installation

1. Clone the repository.

   ```sh
   git clone https://github.com/Praveen3333P/MHN-AI.git
   cd MHN-AI
   ```

2. Create the environment file and fill it in. Every value is documented inline;
   `.env` is gitignored and must never hold committed credentials.

   ```sh
   cp .env.example .env
   ```

   `MHN_SERVICE_TOKEN` is required — the app refuses to start without a token of at least
   32 characters:

   ```sh
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

3. Create the virtual environment and install dependencies.

   ```sh
   uv venv --python 3.12
   uv pip install -r requirements.txt -r requirements-dev.txt
   ```

4. Apply the migrations. Alembic owns the `ai_*` tables only — the Spring-owned tables are
   excluded from autogenerate and are never migrated here.

   ```sh
   alembic upgrade head
   ```

5. Start the stack. The `localdev` profile adds a throwaway Postgres; without it the API and
   worker talk to the real database. S3 and SQS point at real AWS in both cases.

   ```sh
   docker compose --profile localdev up -d
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

Every `/v1` route requires the service token as a bearer credential. `/health` and `/ready`
are unauthenticated so probes keep working.

Submit one or many documents by their `unclassified_files` ids:

```sh
curl -X POST http://localhost:8000/v1/document-processing-runs \
  -H "Authorization: Bearer $MHN_SERVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"document_ids": [101, 102]}'
```

`202 Accepted` returns a `run_id` and a per-document item id. Poll the run for progress —
each item moves through `pending → queued → processing → classifying → extracting →
generating_insights → completed`, or ends at `failed`, `rejected`, or `cancelled`:

```sh
curl http://localhost:8000/v1/document-processing-runs/$RUN_ID \
  -H "Authorization: Bearer $MHN_SERVICE_TOKEN"
```

Read the result for a single document — its detected section, and, when it was moved into
`reports`, the created `reports` id plus the extraction and insights. The document's type
goes in the path, and the route answers only if that is what the document was classified as:

```sh
curl http://localhost:8000/v1/documents/reports/101/ai-result \
  -H "Authorization: Bearer $MHN_SERVICE_TOKEN"
```

`{type}` is one of `reports`, `scans`, `insurance`, `vaccinations`, `prescriptions`. It is
always the section **we detected**, never one the caller declares — every upload arrives
unclassified, so there is nothing to declare at submit time. Each item in the run response
carries a `document_type` telling you which URL to call; asking under the wrong type returns
`409` naming the real section.

The remaining routes are `POST /v1/documents/{type}/{id}/ai-result:retry` (retry a document
that did not complete) and `DELETE /v1/document-processing-runs/{id}` (cancel unfinished
items). Interactive docs are at `/docs`.

A document classified into a non-report section reaches `rejected` with that section as the
reason. That is routing, not a processing error.

### Running the checks

```sh
ruff check app tests
ruff format --check app tests
mypy app
pytest                      # add -m "not integration" to skip the DB-backed tests
```

Integration tests run against a live database inside a rolled-back transaction, with moto
standing in for S3/SQS and a fake provider standing in for the AI calls. No test spends
money and no test writes to Spring-owned tables.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Upload classification into all app sections
- [x] Reports pipeline: extraction, deterministic normalization, insights
- [x] Atomic move into `reports` with the assembled `content` payload
- [x] Parallel workers with bounded concurrency, retries, and a DLQ
- [x] Per-stage models and AI cost/token logging
- [x] Approved-THP age-group ideal-range override (behind `IDEAL_RANGES_ENABLED`, off until
      the Spring parameter tables and the approval predicate are confirmed)
- [x] Reference-range parsing for the shapes labs actually print, and flags for non-numeric
      results (censored, present/absent, qualitative)
- [ ] Additional sections (scans/imaging, prescriptions, insurance) via a section-dispatch
      table — reusing the one worker and one queue, not adding new ones
- [ ] Stale-item reaper for interrupted work
- [ ] Production hardening: managed database, secrets manager, network isolation, CI/CD,
      observability, and DLQ alerting

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Work on your own branch off `main` and open a pull request for review — one branch, one PR
at a time.

1. Branch from `main` (`git checkout -b your-name`)
2. Make the change, and add tests for it
3. Run format, lint, type checks, and the test suite; they must all be clean
4. Commit and push, then open a PR into `main`

Two rules that are easy to trip over:

- **Never issue DDL against Spring-owned tables**, and never use `create_all()`. Alembic
  migrations are for the `ai_*` tables only.
- **Never log report contents, credentials, or prompts.** Failure data is sanitized before
  it reaches the cost logs; there is a test that plants a patient string and asserts it
  never appears.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Proprietary — © MyHealthNotion. All rights reserved. Not licensed for use, copying, or
distribution outside the organisation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
