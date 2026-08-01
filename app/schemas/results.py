"""Response models for a document's AI result and its retry.

The unit is an uploaded document (an ``unclassified_files`` id). ``section_row_id`` is set
only once the document was classified and filed into its section table; the
extraction and insights payloads are the same JSON stored under ``reports.content``.
"""

import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    """The document types addressable in a URL path.

    A **URL vocabulary**, deliberately not the same enum as ``DocumentSection``: ``scans``
    reads better in a path than the section's ``scans_imaging``, and only the types Spring
    routes are addressable. ``app.services.results.SECTION_BY_DOCUMENT_TYPE`` maps one to
    the other.

    The type in a path is always the section a document was **classified as**, never a
    declaration by the caller — every upload lands in ``unclassified_files`` and this
    service detects the section, so at submit time there is no type to state. Declaring it
    as a path parameter makes FastAPI reject an unknown type with 422 before any query runs.
    """

    REPORTS = "reports"
    SCANS = "scans"
    INSURANCE = "insurance"
    VACCINATIONS = "vaccinations"
    PRESCRIPTIONS = "prescriptions"


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
    #: Set when the document was filed into its section table.
    section_row_id: int | None = None
    last_error_code: str | None = None
    #: The section the user uploaded into, or null for a global upload.
    intended_section: str | None = None

    classification: ClassificationResult | None = None
    #: {"results": [...], "report_date": ...} — a REPORT's lab results, once extraction ran.
    extraction: dict[str, Any] | None = None
    #: {"insights": [...], "summary": ..., "disclaimer": ...} — present once insights ran.
    insights: dict[str, Any] | None = None
    #: {"section": ..., "fields": {...}, "flags": [...]} — a NON-report section's fields
    #: (insurance, scans/imaging, vaccinations). Mutually exclusive with ``extraction``:
    #: the two carry different shapes, and which one is populated follows the section.
    #: ``fields`` is whatever that section's ``SectionSpec`` defines, so it differs per
    #: section; ``flags`` carries data-quality notes such as ``dates_out_of_order``.
    section_extraction: dict[str, Any] | None = None


class RetryResponse(BaseModel):
    document_id: int
    item_id: uuid.UUID
    run_id: uuid.UUID
    #: `queued` once re-published; `pending` if publishing failed (the sweep retries it).
    status: str
