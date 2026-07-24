"""Response models for a document's AI result and its retry.

The unit is an uploaded document (an ``unclassified_files`` id). ``reports_id`` is set
only once the document was classified as a report and moved into ``reports``; the
extraction and insights payloads are the same JSON stored under ``reports.content``.
"""

import uuid
from typing import Any

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    section: str = Field(description="The detected MyHealthNotion section, or 'unknown'.")
    title: str
    confidence: float
    reasoning: str | None = None


class DocumentAiResult(BaseModel):
    document_id: int
    item_id: uuid.UUID
    run_id: uuid.UUID
    #: Current lifecycle state of the latest processing item for this document.
    status: str
    #: Set when the document was moved into the reports table.
    reports_id: int | None = None
    last_error_code: str | None = None

    classification: ClassificationResult | None = None
    #: {"results": [...], "report_date": ...} — present once extraction ran.
    extraction: dict[str, Any] | None = None
    #: {"insights": [...], "summary": ..., "disclaimer": ...} — present once insights ran.
    insights: dict[str, Any] | None = None


class RetryResponse(BaseModel):
    document_id: int
    item_id: uuid.UUID
    run_id: uuid.UUID
    #: `queued` once re-published; `pending` if publishing failed (the sweep retries it).
    status: str
