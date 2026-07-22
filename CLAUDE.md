# MHN AI — Claude Project Guide

## Project purpose

Build a FastAPI service for MyHealthNotion's AI-assisted medical-document processing.

**Current sprint scope: reports only.** Implement report auto-classification, structured report extraction, insight generation, parallel worker processing, and AI model/cost logging.

Do not implement scans, imaging, insurance, prescriptions, or multi-document-type routing in this sprint unless explicitly asked.

## Architecture decisions

- FastAPI is the HTTP API and orchestration layer.
- Use **one standard SQS queue** named for report processing. Do not create a queue per pipeline stage or document type.
- Use independently scalable report workers. The API should return `202 Accepted` after persisting and publishing a processing item; it must not perform long-running AI work inline.
- Use PostgreSQL for durable processing-run state, results, audit records, retries, and idempotency.
- Source files remain private in S3. Accept/report S3 object keys; never expose buckets or make objects public.
- Workers may run reports in parallel, but concurrency must be bounded and configured through environment variables.
- SQS is at-least-once: all worker actions must be idempotent.

## Database

The local database is `mhn_ai`.

Base schema source: `D:\mhn-spring\migrations\index.sql`.

Use Alembic migrations for all service-owned tables. Expected tables include:

- `ai_processing_runs`
- `ai_processing_run_items`
- `ai_report_classifications`
- `ai_report_extractions`
- `ai_report_insights`
- `ai_process_logs`

Persist final user-facing output in `reports.content` as JSONB, while retaining detailed/versioned operational data in the `ai_*` tables.

Never make schema changes with `create_all()` in application startup. Use migrations only.

## API contract

Keep routes resource-oriented and versioned:

- `POST /v1/report-processing-runs`
- `GET /v1/report-processing-runs/{run_id}`
- `GET /v1/reports/{report_id}/ai-result`
- `POST /v1/reports/{report_id}/ai-result:retry`
- `DELETE /v1/report-processing-runs/{run_id}`
- `GET /health`
- `GET /ready`

Use Pydantic v2 request and response schemas. Return stable machine-readable error bodies. Do not expose internal stack traces, S3 keys beyond authorized internal usage, prompts, or secrets in API responses.

## Processing lifecycle

Use explicit states for each run item:

`pending -> queued -> processing -> classifying -> extracting -> generating_insights -> completed`

Terminal alternatives: `failed`, `rejected`, `cancelled`.

Rules:

- A duplicate submission must reuse or safely reject the existing active item rather than process the report twice.
- Wrong document types become `rejected` with a clear reason.
- Retry only transient or explicitly retried failures; never overwrite a completed result without `force_reprocess`.
- Send failed messages to a dead-letter queue after the configured retry limit.
- Mark interrupted/incomplete jobs retryable during recovery.

## AI rules

- Require structured model output and validate it with Pydantic before database writes.
- Record model provider, model name, prompt/schema version, input/output tokens, estimated cost, stage duration, outcome, and sanitized failure data in `ai_process_logs`.
- Keep raw model responses only when required for audit/debugging; never log report contents or credentials to stdout.
- Insights must be informational and must not present a diagnosis, emergency instruction, or medical certainty.
- Do not silently repair invalid model output. Record validation failures and retry/fail according to the job policy.

## Report extraction expectations

- Validate S3 object existence, content type, and file size before AI processing.
- Support only the approved PDF/image formats and limits.
- Extract structured lab/report data, including test name, value, unit, reference range, observed date, and source context when available.
- Normalize compatible units and calculate abnormal/out-of-range flags deterministically; do not ask the LLM to perform arithmetic that application code can verify.
- Store extraction and insights separately, then assemble the final `reports.content` payload.

## Code standards

- Python 3.12+, FastAPI, SQLAlchemy 2.x, Alembic, Pydantic v2.
- Keep HTTP handlers thin; place business logic in services and infrastructure code behind interfaces.
- Use dependency injection for database sessions, S3 clients, SQS clients, and AI providers.
- Use timezone-aware UTC timestamps.
- Use UUIDs for processing run identifiers.
- Configuration belongs in environment variables and `.env.example`; never commit secrets or real credentials.
- Prefer explicit types and small functions. Avoid global mutable state.
- Do not use FastAPI `BackgroundTasks` for durable report processing.

## Testing and verification

Before declaring a change complete:

1. Run format, lint, and type checks configured by the repository.
2. Add/adjust unit tests for pure logic.
3. Add integration tests for database, idempotency, API status transitions, and mocked S3/SQS/AI boundaries.
4. Run an end-to-end test against a representative report fixture.
5. Verify that retries do not duplicate results or cost logs.

## Working rules

- Inspect existing code and migrations before changing them.
- Preserve unrelated user changes.
- Do not create AWS resources, execute production migrations, or use real patient data without explicit approval.
- If a requirement conflicts with this document, ask for clarification rather than guessing.
