"""Text extraction from a source document: text layer first, OCR as the fallback.

Per page, in order of preference:

1. **pdfplumber** (MIT) reads the embedded text layer *sorted by position on the page*.
   Exact, free, and how most insurance policies and lab reports arrive.
2. **pypdfium2** (BSD-3/Apache-2.0) reads the same layer in storage order, for any page
   pdfplumber returned nothing for — either because it could not parse the file at all,
   or because that one page carries no text of its own.
3. **Image-only page, unusable text, or an image file** — rasterise at ``RASTER_DPI`` and
   OCR with Tesseract, keeping the word-level confidence.

Calling pypdfium2 a fallback undersells it: it also opens every document and drives the
page loop — ``len(pdf)`` is the page count, and pdfplumber's output is indexed into that —
and it rasterises every page that reaches step 3. It is the file handling throughout;
pdfplumber is the reading.

Which makes these two text engines, not one: pdfplumber reads through **pdfminer.six**,
pypdfium2 through **PDFium**. pdfplumber requires pypdfium2 only as its rasteriser
(``display.py``), never for ``extract_text``, so carrying both does not shrink the
dependency tree. The justification is the accuracy measured below — not consolidation.

The distinction between 1 and 2 is not cosmetic. A PDF stores characters in whatever
order the generating software emitted them, which need not be the order a person reads
them. Some lab reports store pages bottom-to-top, so an unsorted read returns the
impression first and the patient details last; worse, a two-column header can arrive as
five labels followed by five values, leaving the model to pair ``NAME`` with
``Mr.DINESH`` by position alone. Sorting by coordinate removes the guess.

Measured on two real lab reports of deliberately different layout — Dr.Remedies Labs
(13 pages, ruled tables) and Thyrocare (9 pages, unruled columns with ranges stacked
under the test name). 130 fields transcribed by reading the rendered pages, scoring a
value only when it sits beside its own test name:

    engine                  names      values     ranges   all three
    pdfplumber+pypdfium2   130/130    130/130    110/110    130/130
    pypdfium2 alone        130/130    127/130    105/110    122/130
    pymupdf                130/130     40/130     31/110     37/130

PyMuPDF, which this replaces, emits whole rows reversed — every character present, the
pairing destroyed. pypdfium2 alone loses three values and five multi-line reference
ranges to storage order, among them a Vitamin B12 of 185 against a range of 211-946: a
deficiency that vanishes without trace. ``docs/extraction-cost-options.md`` benchmarks
speed and character counts and prefers pypdfium2 alone; that measures a different axis
and neither result contradicts the other. The ~1s/document pdfplumber costs is small
against a model call.

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
import re
from dataclasses import dataclass, field

import pdfplumber
import pypdfium2 as pdfium
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
#: A page with fewer *informative* characters than this is treated as image-only. Digital
#: pages that are genuinely near-empty cost one wasted OCR attempt, the safer error.
MIN_TEXT_CHARS = 15

_IMAGE_CONTENT_TYPES = frozenset({"image/jpeg", "image/png"})

#: Block-element and box-drawing glyphs — how a barcode font renders as text. A lab
#: report whose only real text is the blank form plus a barcode runs to well over a
#: thousand characters and says nothing about the patient, so counting raw characters
#: would wave it past MIN_TEXT_CHARS and skip the OCR it needs.
_BARCODE_GLYPHS = re.compile("[─-▟]+")
_ALNUM = re.compile(r"[^\W_]")
#: ``layout=True`` pads every line to the page's full width so columns line up visually.
#: Almost none of that padding informs anything — what a model needs is the *fact* of a
#: column break, not forty spaces proving it. Collapsing runs to three cut the extracted
#: text by 58% on the sample documents with no field lost; left in, this stage would
#: send roughly three times the tokens it needs to.
_RUN_OF_SPACES = re.compile(r" {3,}")
#: A page is judged scrambled when this share of its non-empty lines carry one
#: informative character or fewer. Measured on real documents the separation is stark:
#: 52% on a scrambled page against 0% on a healthy one, so the threshold is not delicate.
_SCRAMBLE_RATIO = 0.25
_MIN_SHORT_LINES = 8

#: Three spaces — the same column marker ``_collapse_padding`` leaves on the text path,
#: so a page reads the same to the model whichever path produced it.
_COLUMN_GAP = "   "
#: A horizontal gap wider than this many median glyph heights reads as a column break
#: rather than a word space. Deriving it from the glyph height rather than a pixel count
#: keeps it correct at any ``RASTER_DPI``; one em sits comfortably above the ~0.3em of a
#: space and below any real gutter.
_COLUMN_GAP_EMS = 1.0


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


def extract_text(document: DocumentPayload, *, allow_ocr: bool = True) -> ExtractedText:
    """Read a document to text. Raises ``TextExtractionError`` if it cannot be opened.

    ``allow_ocr=False`` reads the embedded text layer and nothing else: a page that would
    have been OCR'd is marked ``"skipped"`` and contributes no text. For a caller that
    cannot use an OCR'd page anyway — ``prescriptions._verify_against_document`` will not
    reject a drug name on OCR output — running the engine and discarding the result is a
    full Tesseract pass bought for nothing, and a photographed prescription is several
    seconds of it.

    Crucially the pages are *skipped*, not read poorly. A page whose text layer came back
    scrambled is exactly where a text-only read would be most tempting and least
    trustworthy, so it is dropped rather than passed on with the OCR that would have
    replaced it.
    """
    if not document.data:
        raise TextExtractionError("Source file is empty")

    if document.content_type in _IMAGE_CONTENT_TYPES:
        if not allow_ocr:
            # An image has no text layer to read; OCR is the only path and it is barred.
            return ExtractedText(
                text="",
                page_count=1,
                ocr_pages=0,
                engine=None,
                mean_confidence=None,
                pages=[PageText(0, "skipped", 0, None)],
            )
        return _from_image(document.data)
    return _from_pdf(document.data, allow_ocr=allow_ocr)


# --- usability of a text layer ----------------------------------------------


def _strip_barcodes(text: str) -> str:
    """Drop barcode glyph runs. They are never content and always cost tokens."""
    return _BARCODE_GLYPHS.sub(" ", text)


def _collapse_padding(text: str) -> str:
    """Keep the column separation layout mode gives, drop the padding that proves it.

    Three spaces still reads unambiguously as a column break, to a person and to a
    model, and costs a fraction of the forty needed to reach the real x-position.
    """
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
    Rasterising and OCR'ing recovers it, because OCR reads what the page looks like
    rather than how it was built.

    One-character lines are rare in ordinary text, which makes their share a cheap and
    specific signal.
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


def _from_pdf(data: bytes, *, allow_ocr: bool = True) -> ExtractedText:
    layout_pages = _layout_pages(data)

    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as exc:
        raise TextExtractionError(f"Could not open PDF: {exc}") from exc

    parts: list[str] = []
    pages: list[PageText] = []
    confidences: list[float] = []
    ocr_pages = 0

    try:
        for index in range(len(pdf)):
            page = pdf[index]
            try:
                best = ""
                if layout_pages is not None and index < len(layout_pages):
                    best = layout_pages[index]

                if _informative(best) < MIN_TEXT_CHARS:
                    best = _pypdfium_text(page)

                # OCR when there is no usable text layer, and also when the text came
                # back scrambled: every character present but the words destroyed, which
                # a character count alone cannot distinguish from a healthy page.
                if _informative(best) < MIN_TEXT_CHARS or _looks_scrambled(best):
                    if not allow_ocr:
                        # Skipped rather than kept: what is here is either too little to
                        # read or scrambled, and the caller barred the engine that would
                        # have fixed it. Passing it on would hand over the worst text on
                        # the document dressed as a clean text layer.
                        pages.append(PageText(index, "skipped", 0, None))
                        continue

                    if ocr_pages >= MAX_OCR_PAGES:
                        pages.append(PageText(index, "skipped", 0, None))
                        logger.warning(
                            "OCR page cap (%d) reached; skipping page %d", MAX_OCR_PAGES, index
                        )
                        continue

                    text, confidence = _ocr_pdf_page(page, index)
                    ocr_pages += 1
                    # A poor scan can read worse than the flawed text layer it replaces,
                    # so keep whichever carries more that a reader could use.
                    if _informative(text) > _informative(best):
                        if text:
                            parts.append(text)
                        if confidence is not None:
                            confidences.append(confidence)
                        pages.append(PageText(index, "ocr", len(text), confidence))
                        continue

                if best:
                    parts.append(best)
                    pages.append(PageText(index, "text", len(best), None))
                else:
                    pages.append(PageText(index, "skipped", 0, None))
            finally:
                page.close()
    finally:
        pdf.close()

    return ExtractedText(
        text="\n\n".join(part for part in parts if part).strip(),
        page_count=len(pages),
        ocr_pages=ocr_pages,
        engine="tesseract" if ocr_pages else None,
        mean_confidence=_mean(confidences),
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


def _ocr_pdf_page(page: pdfium.PdfPage, index: int) -> tuple[str, float | None]:
    """OCR one rasterised page. A single bad page must not lose the whole document."""
    try:
        image = page.render(scale=RASTER_DPI / 72).to_pil().convert("RGB")
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

    Word-level output is used rather than plain ``image_to_string`` for two reasons. The
    confidence comes back with it: a low-confidence read is the signal that a field was
    missed because the scan was poor, not because the model failed. And Tesseract numbers
    every word by block, paragraph and line, which is what lets the words be reassembled
    into the lines and columns the page actually had.

    That reassembly is not decoration. Joining every word with a single space would hand
    the model a lab report as one undivided stream — name, value, unit and range from one
    row running straight into the next — which is exactly the pairing the text path
    exists to preserve. The OCR path is live (a scrambled-glyph report in the sample set
    OCRs all ten pages), so it has to preserve as much structure as the path it replaces.
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    indices = [index for index, word in enumerate(data["text"]) if word.strip()]
    if not indices:
        return "", None

    heights = sorted(int(data["height"][index]) for index in indices)
    gap_threshold = heights[len(heights) // 2] * _COLUMN_GAP_EMS

    confidences: list[float] = []
    lines: list[str] = []
    current: list[str] = []
    current_key: tuple[int, int, int] | None = None
    previous_right = 0

    for index in indices:
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        left = int(data["left"][index])

        if key != current_key:
            if current:
                lines.append("".join(current))
            current = []
            current_key = key
        elif left - previous_right >= gap_threshold:
            current.append(_COLUMN_GAP)
        else:
            current.append(" ")

        current.append(data["text"][index])
        previous_right = left + int(data["width"][index])

        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        # Tesseract reports -1 for words it did not score; excluded rather than
        # counted as zero, which would drag the mean down misleadingly.
        if confidence >= 0:
            confidences.append(confidence / 100.0)

    if current:
        lines.append("".join(current))

    return "\n".join(lines), _mean(confidences)


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


# Applied on import, because ``pytesseract`` reads this as a module global at call time
# and every path that OCRs comes through here. Left to the caller it is documented
# behaviour that silently does nothing until someone remembers to invoke it — and the
# symptom, a scanned document failing while digital ones work, points nowhere near the
# cause. Doing it here also means ``tesseract_available()`` answers about the binary the
# module will actually run.
configure_tesseract_from_env()
