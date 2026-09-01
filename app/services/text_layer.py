"""Read a document's embedded text layer. No OCR, and deliberately none.

**One caller: the prescription name guard** (``prescriptions._verify_against_document``).
That guard deletes a prescribed medicine when the document's own text does not support the
name the model returned, so the text it rests on has to be at least as trustworthy as the
model. An OCR'd page never was — which is why the guard has always passed
``allow_ocr=False`` — and now there is no OCR to pass it to.

Section extraction used to be the other caller. It sends the **document** to the vision
model now (2026-09-01), the way reports and prescriptions always have, so the OCR half of
this module had exactly one consumer left and that consumer never wanted it. What went
with it: Tesseract, ``pytesseract``, rasterisation, word-level confidence, the OCR page
cap, and the six tests that skipped on every machine without the binary.

Per page, in order of preference:

1. **pdfplumber** (MIT) reads the embedded text layer *sorted by position on the page*.
2. **pypdfium2** (BSD-3/Apache-2.0) reads the same layer in storage order, for any page
   pdfplumber returned nothing for — either because it could not parse the file at all, or
   because that one page carries no text of its own.
3. **Anything else — an image-only page, unusable text, or an image file** — is
   ``skipped``. It contributes no text and is *counted*, which is what lets the guard say
   "these names were not checked" rather than checking them against nothing.

The distinction between 1 and 2 is not cosmetic. A PDF stores characters in whatever order
the generating software emitted them, which need not be the order a person reads them.
Sorting by coordinate removes the guess. Measured on two real lab reports of deliberately
different layout, 130 fields, scoring a value only when it sits beside its own test name:

    engine                  names      values     ranges   all three
    pdfplumber+pypdfium2   130/130    130/130    110/110    130/130
    pypdfium2 alone        130/130    127/130    105/110    122/130
    pymupdf                130/130     40/130     31/110     37/130

These are **two text engines**, not one: pdfplumber reads through pdfminer.six, pypdfium2
through PDFium. pdfplumber needs pypdfium2 only as its rasteriser, so carrying both does
not shrink the dependency tree — the justification is the accuracy above.

Extracted text is used and discarded; it is never persisted. Only the small non-PII
provenance from ``as_metadata()`` is kept.
"""

import io
import logging
import re
from dataclasses import dataclass, field

import pdfplumber
import pypdfium2 as pdfium

from app.integrations.ai.base import DocumentPayload

logger = logging.getLogger(__name__)

#: A page with fewer *informative* characters than this has no usable text layer.
MIN_TEXT_CHARS = 15

_IMAGE_CONTENT_TYPES = frozenset({"image/jpeg", "image/png"})

#: Block-element and box-drawing glyphs — how a barcode font renders as text. A lab report
#: whose only real text is the blank form plus a barcode runs to well over a thousand
#: characters and says nothing, so counting raw characters would wave it past
#: MIN_TEXT_CHARS.
_BARCODE_GLYPHS = re.compile("[─-▟]+")
_ALNUM = re.compile(r"[^\W_]")
#: ``layout=True`` pads every line to the page's full width so columns line up visually.
#: Almost none of that padding informs anything — what a reader needs is the *fact* of a
#: column break, not forty spaces proving it. Collapsing runs to three cut the extracted
#: text by 58% on the sample documents with no field lost.
_RUN_OF_SPACES = re.compile(r" {3,}")
#: A page is judged scrambled when this share of its non-empty lines carry one informative
#: character or fewer. Measured on real documents the separation is stark: 52% on a
#: scrambled page against 0% on a healthy one, so the threshold is not delicate.
_SCRAMBLE_RATIO = 0.25
_MIN_SHORT_LINES = 8


class TextExtractionError(Exception):
    """The document could not be opened at all. Permanent for this file."""


@dataclass(frozen=True)
class PageText:
    index: int
    #: "text" (a usable embedded layer) | "skipped" (none, or scrambled beyond use)
    method: str
    chars: int


@dataclass(frozen=True)
class ExtractedText:
    text: str
    page_count: int
    pages: list[PageText] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def skipped_pages(self) -> int:
        return sum(1 for page in self.pages if page.method == "skipped")

    def as_metadata(self) -> dict[str, object]:
        """The non-PII provenance stored alongside an extraction. Never the text."""
        return {
            "page_count": self.page_count,
            "skipped_pages": self.skipped_pages,
            "chars": self.chars,
        }


def extract_text(document: DocumentPayload) -> ExtractedText:
    """Read a document's text layer. Raises ``TextExtractionError`` if it cannot be opened.

    A page with no usable layer is **skipped**, never guessed at. The one caller rejects
    model output on the strength of this text, so a page read badly is worse to it than a
    page not read at all.
    """
    if not document.data:
        raise TextExtractionError("Source file is empty")

    if document.content_type in _IMAGE_CONTENT_TYPES:
        # An image has no text layer. Reported as one skipped page rather than as an
        # error: "this could not be checked" is a real answer and the guard acts on it.
        return ExtractedText(text="", page_count=1, pages=[PageText(0, "skipped", 0)])

    return _from_pdf(document.data)


# --- usability of a text layer ----------------------------------------------


def _strip_barcodes(text: str) -> str:
    """Drop barcode glyph runs. They are never content and always cost characters."""
    return _BARCODE_GLYPHS.sub(" ", text)


def _collapse_padding(text: str) -> str:
    """Keep the column separation layout mode gives, drop the padding that proves it."""
    lines = []
    for raw in text.splitlines():
        line = _RUN_OF_SPACES.sub("   ", raw).rstrip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _informative(text: str) -> int:
    """Letters and digits only, barcode glyphs excluded."""
    return len(_ALNUM.findall(_strip_barcodes(text)))


def _looks_scrambled(text: str) -> bool:
    """Whether position-sorted reading broke a block into single characters.

    Some PDFs draw a header by placing each glyph independently. Sorting by coordinate
    then emits one character per line — ``P``, ``A``, ``a``, ``g``, ``t`` — which is
    "Patient Name" and "Age" interleaved: every character present, every word destroyed.

    There is no OCR to escalate to any more, so such a page is **skipped**. That is the
    right answer for the one caller: a guard that rejects a drug name because the page's
    letters arrived shuffled would delete real medicines.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < _MIN_SHORT_LINES:
        return False
    short = sum(1 for line in lines if _informative(line) <= 1)
    return short >= _MIN_SHORT_LINES and short / len(lines) >= _SCRAMBLE_RATIO


# --- PDF --------------------------------------------------------------------


def _layout_pages(data: bytes) -> list[str] | None:
    """Position-sorted text per page, or None when pdfplumber cannot parse the file.

    Returning None rather than raising keeps a parser failure from failing the document:
    pypdfium2 then reads every page instead.
    """
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return [
                _collapse_padding(_strip_barcodes(page.extract_text(layout=True) or ""))
                for page in pdf.pages
            ]
    except Exception as exc:
        logger.warning("pdfplumber could not parse this PDF (%s); using pypdfium2", exc)
        return None


def _from_pdf(data: bytes) -> ExtractedText:
    layout_pages = _layout_pages(data)

    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as exc:
        raise TextExtractionError(f"Could not open PDF: {exc}") from exc

    parts: list[str] = []
    pages: list[PageText] = []

    try:
        for index in range(len(pdf)):
            page = pdf[index]
            try:
                best = ""
                if layout_pages is not None and index < len(layout_pages):
                    best = layout_pages[index]

                if _informative(best) < MIN_TEXT_CHARS:
                    best = _pypdfium_text(page)

                if _informative(best) < MIN_TEXT_CHARS or _looks_scrambled(best):
                    pages.append(PageText(index, "skipped", 0))
                    continue

                parts.append(best)
                pages.append(PageText(index, "text", len(best)))
            finally:
                page.close()
    finally:
        pdf.close()

    return ExtractedText(
        text="\n\n".join(part for part in parts if part).strip(),
        page_count=len(pages),
        pages=pages,
    )


def _pypdfium_text(page: pdfium.PdfPage) -> str:
    """The page's text layer in storage order — the fallback when pdfplumber cannot."""
    try:
        textpage = page.get_textpage()
    except Exception:
        return ""
    try:
        return _strip_barcodes(textpage.get_text_range() or "").strip()
    finally:
        textpage.close()
