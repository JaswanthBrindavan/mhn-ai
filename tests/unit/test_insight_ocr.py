"""Text extraction: text layer first, Tesseract OCR for image-only pages.

The OCR cases build an image-only PDF by rasterising a text one — that is what a scanner
or a phone photo produces, and it is the only way to exercise the fallback without
committing a binary fixture.
"""

import fitz
import pytest

from app.insights.ocr import (
    MIN_TEXT_CHARS,
    ExtractedText,
    PageText,
    TextExtractionError,
    extract_text,
    tesseract_available,
)
from app.integrations.ai.base import DocumentPayload

needs_tesseract = pytest.mark.skipif(
    not tesseract_available(), reason="Tesseract binary not installed"
)

SAMPLE = (
    "ISLAMABAD DIAGNOSTIC CENTRE\n"
    "Chest X-Ray PA View\n"
    "Date 31/08/2021\n"
    "Impression: Minimal bilateral apical pleural thickening."
)


def _text_pdf(body: str = SAMPLE, pages: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), body, fontsize=11)
    data: bytes = doc.tobytes()
    doc.close()
    return data


def _scanned_pdf(body: str = SAMPLE, dpi: int = 200) -> bytes:
    """A text PDF rasterised to images — no text layer, so OCR is the only way in."""
    src = fitz.open(stream=_text_pdf(body), filetype="pdf")
    out = fitz.open()
    for page in src:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=pixmap.tobytes("png"))
    data: bytes = out.tobytes()
    src.close()
    out.close()
    return data


def _payload(data: bytes, content_type: str = "application/pdf") -> DocumentPayload:
    return DocumentPayload(data=data, content_type=content_type, filename="doc.pdf")


# --- text layer -------------------------------------------------------------


def test_digital_pdf_uses_the_text_layer_and_never_ocrs():
    result = extract_text(_payload(_text_pdf()))

    assert "ISLAMABAD DIAGNOSTIC CENTRE" in result.text
    assert result.used_ocr is False
    assert result.ocr_pages == 0
    assert result.engine is None  # no engine ran, so none is recorded
    assert result.mean_confidence is None
    assert [p.method for p in result.pages] == ["text"]


def test_every_page_is_read():
    result = extract_text(_payload(_text_pdf(pages=3)))

    assert result.page_count == 3
    assert result.text.count("Chest X-Ray") == 3


def test_metadata_carries_provenance_but_never_the_text():
    metadata = extract_text(_payload(_text_pdf())).as_metadata()

    assert metadata["used_ocr"] is False
    assert metadata["page_count"] == 1
    assert isinstance(metadata["chars"], int)
    # The extracted text is deliberately absent — it is never persisted.
    assert "text" not in metadata


# --- OCR fallback -----------------------------------------------------------


@needs_tesseract
def test_scanned_pdf_falls_back_to_ocr():
    result = extract_text(_payload(_scanned_pdf()))

    assert result.used_ocr is True
    assert result.ocr_pages == 1
    assert result.engine == "tesseract"
    assert [p.method for p in result.pages] == ["ocr"]


@needs_tesseract
def test_ocr_recovers_the_documents_content():
    """OCR is lossy, so assert on the fields that matter rather than an exact match."""
    result = extract_text(_payload(_scanned_pdf()))

    assert "DIAGNOSTIC" in result.text.upper()
    assert "31/08/2021" in result.text


@needs_tesseract
def test_ocr_reports_a_usable_confidence():
    result = extract_text(_payload(_scanned_pdf()))

    assert result.mean_confidence is not None
    assert 0.0 <= result.mean_confidence <= 1.0
    # A clean 200-DPI render of printed text should score well; a much lower figure
    # means the rasterise/OCR settings have regressed.
    assert result.mean_confidence > 0.5


@needs_tesseract
def test_a_page_with_a_thin_text_layer_is_treated_as_image_only():
    """Below MIN_TEXT_CHARS the 'layer' is page furniture, not content — OCR it."""
    result = extract_text(_payload(_text_pdf(body="x" * (MIN_TEXT_CHARS - 5))))

    assert result.used_ocr is True


# --- failures ---------------------------------------------------------------


def test_empty_file_is_rejected():
    with pytest.raises(TextExtractionError, match="empty"):
        extract_text(_payload(b""))


def test_unopenable_pdf_is_rejected():
    with pytest.raises(TextExtractionError, match="Could not open PDF"):
        extract_text(_payload(b"%PDF-1.4 this is not really a pdf"))


def test_unopenable_image_is_rejected():
    with pytest.raises(TextExtractionError, match="Could not open image"):
        extract_text(_payload(b"not an image", content_type="image/png"))


def test_a_pdf_with_no_content_yields_empty_text_not_an_error():
    """An empty page is a readable document that happens to say nothing. The caller
    decides what to do; extraction does not invent a failure."""
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()

    result = extract_text(_payload(data))
    assert result.text == ""


# --- shape ------------------------------------------------------------------


def test_chars_reflects_the_extracted_text():
    result = ExtractedText(
        text="abcd",
        page_count=1,
        ocr_pages=0,
        engine=None,
        mean_confidence=None,
        pages=[PageText(0, "text", 4, None)],
    )
    assert result.chars == 4
    assert result.used_ocr is False
