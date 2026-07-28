"""Text extraction from a source document: text layer first, OCR as the fallback.

Per page, in order of preference:

1. **Embedded text layer** (a digitally generated PDF) — read it with PyMuPDF. Exact,
   free, and instant. Most insurance policies and lab reports arrive this way.
2. **Image-only page or an image file** — rasterise at ``RASTER_DPI`` and OCR with
   Tesseract, keeping the word-level confidence.

The distinction matters for accuracy, not just cost: a text layer is what the document
*says*, while OCR is a best guess at what it looks like. Where a text layer exists it is
always the better source, so it is never overridden.

Measured on the sample documents — each one first as issued, then rasterised to force
the OCR path:

    Chest X-Ray               1,424 chars in  3 ms  ->  1,358 chars, confidence 0.91
    Insurance receipt         1,669 chars in  9 ms  ->  1,467 chars, confidence 0.91
    Vaccination certificate   1,251 chars in 48 ms  ->  1,352 chars, confidence 0.76

Every sample is digital, so in practice the text layer handles them and OCR never runs.
It exists for the phone photo of a vaccination card.

**Why extract text at all**, rather than hand the file to the model as the report
pipeline does: the token cost becomes bounded by the text rather than the page count,
and — more usefully — a failure becomes attributable. The engine, page counts and mean
confidence are stored beside every extraction, so a missing field can be traced to a bad
scan instead of blamed on the model. The cost of that bet is real: OCR becomes the
accuracy ceiling, because whatever Tesseract drops the model never sees.

Extracted text is handed to the model and then discarded — it is never persisted. Only
the small non-PII metadata from ``as_metadata()`` is kept, which is what lets a reviewer
tell "the model missed it" from "OCR never saw it".

OCR needs the **Tesseract binary**, which ``pytesseract`` only binds to. Without it
digital PDFs still work — a text layer needs no binary — and scanned ones fail.
``TESSERACT_CMD`` overrides the binary path when it is not on ``PATH``.

Hard caps bound the work so a pathological upload cannot hold an SQS message past its
visibility timeout.
"""

import io
import logging
import os
from dataclasses import dataclass, field

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from app.integrations.ai.base import DocumentPayload

logger = logging.getLogger(__name__)

#: OCR is the expensive path; a text layer is not, so only OCR is capped.
MAX_OCR_PAGES = 30
#: Rasterisation DPI. 200 is the accuracy/speed knee for printed medical documents:
#: 150 starts losing small print (footnote exclusions, dosage tables), 300 roughly
#: doubles the work for little gain.
RASTER_DPI = 200
#: A page with fewer characters than this is treated as image-only. Digital pages that
#: are genuinely near-empty cost one wasted OCR attempt, which is the safer error.
MIN_TEXT_CHARS = 15

_IMAGE_CONTENT_TYPES = frozenset({"image/jpeg", "image/png"})


class TextExtractionError(Exception):
    """The document could not be read at all. Permanent for this file."""


@dataclass(frozen=True)
class PageText:
    index: int
    #: "text" (embedded layer) | "ocr" | "skipped" (past the OCR cap)
    method: str
    chars: int
    confidence: float | None


@dataclass(frozen=True)
class ExtractedText:
    text: str
    page_count: int
    ocr_pages: int
    #: None when every page had a text layer, so no OCR engine ran.
    engine: str | None
    #: Mean Tesseract word confidence across OCR'd pages, 0-1. None when no OCR ran.
    mean_confidence: float | None
    pages: list[PageText] = field(default_factory=list)

    @property
    def used_ocr(self) -> bool:
        return self.ocr_pages > 0

    @property
    def chars(self) -> int:
        return len(self.text)

    def as_metadata(self) -> dict[str, object]:
        """The non-PII provenance stored alongside an extraction. Never the text."""
        return {
            "page_count": self.page_count,
            "ocr_pages": self.ocr_pages,
            "used_ocr": self.used_ocr,
            "engine": self.engine,
            "mean_confidence": self.mean_confidence,
            "chars": self.chars,
        }


def extract_text(document: DocumentPayload) -> ExtractedText:
    """Read a document to text. Raises ``TextExtractionError`` if it cannot be opened."""
    if not document.data:
        raise TextExtractionError("Source file is empty")

    if document.content_type in _IMAGE_CONTENT_TYPES:
        return _from_image(document.data)
    return _from_pdf(document.data)


# --- PDF --------------------------------------------------------------------


def _from_pdf(data: bytes) -> ExtractedText:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise TextExtractionError(f"Could not open PDF: {exc}") from exc

    parts: list[str] = []
    pages: list[PageText] = []
    confidences: list[float] = []
    ocr_pages = 0

    matrix = fitz.Matrix(RASTER_DPI / 72, RASTER_DPI / 72)
    try:
        for index, page in enumerate(doc):
            layer = page.get_text("text").strip()
            if len(layer) >= MIN_TEXT_CHARS:
                parts.append(layer)
                pages.append(PageText(index, "text", len(layer), None))
                continue

            if ocr_pages >= MAX_OCR_PAGES:
                pages.append(PageText(index, "skipped", 0, None))
                logger.warning("OCR page cap (%d) reached; skipping page %d", MAX_OCR_PAGES, index)
                continue

            text, confidence = _ocr_pdf_page(page, matrix, index)
            ocr_pages += 1
            if text:
                parts.append(text)
            if confidence is not None:
                confidences.append(confidence)
            pages.append(PageText(index, "ocr", len(text), confidence))
    finally:
        doc.close()

    return ExtractedText(
        text="\n\n".join(part for part in parts if part).strip(),
        page_count=len(pages),
        ocr_pages=ocr_pages,
        engine="tesseract" if ocr_pages else None,
        mean_confidence=_mean(confidences),
        pages=pages,
    )


def _ocr_pdf_page(page: fitz.Page, matrix: fitz.Matrix, index: int) -> tuple[str, float | None]:
    """OCR one rasterised page. A single bad page must not lose the whole document."""
    try:
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        return _ocr_image(image)
    except Exception as exc:
        logger.warning("OCR failed on page %d: %s", index, exc)
        return "", None


# --- image ------------------------------------------------------------------


def _from_image(data: bytes) -> ExtractedText:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise TextExtractionError(f"Could not open image: {exc}") from exc

    try:
        text, confidence = _ocr_image(image)
    except Exception as exc:
        raise TextExtractionError(f"OCR failed on image: {exc}") from exc

    text = text.strip()
    return ExtractedText(
        text=text,
        page_count=1,
        ocr_pages=1,
        engine="tesseract",
        mean_confidence=confidence,
        pages=[PageText(0, "ocr", len(text), confidence)],
    )


# --- Tesseract --------------------------------------------------------------


def _ocr_image(image: Image.Image) -> tuple[str, float | None]:
    """Run Tesseract, returning its text and mean word confidence (0-1).

    Word-level output is used rather than plain ``image_to_string`` so the confidence
    comes back too: a low-confidence read is the signal that a field was missed because
    the scan was poor, not because the model failed.
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words: list[str] = []
    confidences: list[float] = []

    for word, raw_confidence in zip(data["text"], data["conf"], strict=False):
        if not word.strip():
            continue
        words.append(word)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            continue
        # Tesseract reports -1 for words it did not score; excluded rather than
        # counted as zero, which would drag the mean down misleadingly.
        if confidence >= 0:
            confidences.append(confidence / 100.0)

    return " ".join(words), _mean(confidences)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def tesseract_available() -> bool:
    """Whether the Tesseract binary is installed. Used to skip OCR-dependent tests."""
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return True


def configure_tesseract_from_env() -> None:
    """Honour ``TESSERACT_CMD`` when the binary is not on PATH (common on Windows)."""
    command = os.getenv("TESSERACT_CMD")
    if command:
        pytesseract.pytesseract.tesseract_cmd = command
