"""Reading a PDF's embedded text layer. There is no OCR path any more.

The seven Tesseract-dependent cases that used to live here went with it. They were
`@needs_tesseract`-skipped on every machine without the binary — which is every developer
machine here — so they contributed the suite's six permanent skips and never once ran
locally. What is left runs everywhere.

Fixtures are written with reportlab (BSD, dev-only). Neither pdfplumber nor pypdfium2 can
*create* a PDF — they read and render — so something has to author the test document.
"""

import io

import pypdfium2 as pdfium
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.integrations.ai.base import DocumentPayload
from app.services.text_layer import (
    MIN_TEXT_CHARS,
    ExtractedText,
    PageText,
    TextExtractionError,
    extract_text,
)
from tests.support.pdfs import TWO_COLUMN_GAP, text_pdf, two_column_pdf

SAMPLE = (
    "ISLAMABAD DIAGNOSTIC CENTRE\n"
    "Chest X-Ray PA View\n"
    "Date 31/08/2021\n"
    "Impression: Minimal bilateral apical pleural thickening."
)


def _blank_pdf() -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _rasterise(data: bytes, dpi: int = 200) -> bytes:
    """The same page with its text layer destroyed — what a scan or a phone photo is."""
    source = pdfium.PdfDocument(data)
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    try:
        for index in range(len(source)):
            page = source[index]
            image = page.render(scale=dpi / 72).to_pil()
            page.close()
            frame = io.BytesIO()
            image.save(frame, format="PNG")
            frame.seek(0)
            from reportlab.lib.utils import ImageReader

            pdf.drawImage(ImageReader(frame), 0, 0, width=A4[0], height=A4[1])
            pdf.showPage()
    finally:
        source.close()
    pdf.save()
    return buffer.getvalue()


def _payload(data: bytes, content_type: str = "application/pdf") -> DocumentPayload:
    return DocumentPayload(data=data, content_type=content_type, filename="doc.pdf")


def test_a_digital_pdf_is_read_from_its_text_layer():
    result = extract_text(_payload(text_pdf(SAMPLE)))

    assert "ISLAMABAD DIAGNOSTIC CENTRE" in result.text
    assert "Minimal bilateral apical pleural thickening" in result.text
    assert result.page_count == 1
    assert [page.method for page in result.pages] == ["text"]
    assert result.skipped_pages == 0


def test_every_page_is_read():
    result = extract_text(_payload(text_pdf(SAMPLE, pages=3)))

    assert result.page_count == 3
    assert [page.method for page in result.pages] == ["text"] * 3


def test_columns_survive_the_read():
    """The whole reason pdfplumber leads: a label and its value must stay on one line.

    Storage order is the order the generator emitted characters, which on a two-column
    header can be five labels followed by five values — every character present, the
    pairing destroyed. Position-sorted reading is what prevents that.
    """
    rows = [("Patient Name", "Mr DINESH"), ("Age", "42 Years"), ("Sex", "Male")]
    result = extract_text(_payload(two_column_pdf(rows)))

    for label, value in rows:
        line = next(ln for ln in result.text.splitlines() if label in ln)
        assert value in line, f"{label!r} lost its value to storage order"
    assert TWO_COLUMN_GAP  # the fixture really does place them apart


def test_a_page_with_no_text_layer_is_skipped_not_guessed_at():
    """A rasterised page is exactly what OCR used to rescue. There is nothing to rescue
    it with now, and inventing text for it would be far worse than reporting none."""
    result = extract_text(_payload(_rasterise(text_pdf(SAMPLE))))

    assert result.text == ""
    assert [page.method for page in result.pages] == ["skipped"]
    assert result.skipped_pages == 1


def test_an_image_file_is_one_skipped_page_rather_than_an_error():
    """ "Could not be checked" is a real answer the prescription guard acts on; a raised
    error would fail a document that is perfectly fine."""
    result = extract_text(_payload(b"\x89PNG\r\n\x1a\n" + b"0" * 64, "image/png"))

    assert result.text == ""
    assert result.page_count == 1
    assert result.skipped_pages == 1


def test_metadata_carries_provenance_but_never_the_text():
    meta = extract_text(_payload(text_pdf(SAMPLE))).as_metadata()

    assert meta == {"page_count": 1, "skipped_pages": 0, "chars": meta["chars"]}
    assert isinstance(meta["chars"], int) and meta["chars"] > 0
    # The one property that matters: no page content in what gets stored.
    assert not any("DINESH" in str(v) or "Impression" in str(v) for v in meta.values())


def test_a_thin_text_layer_counts_as_no_text_layer():
    """Below MIN_TEXT_CHARS of *informative* characters there is nothing to trust."""
    result = extract_text(_payload(text_pdf("20cm")))

    assert len("20cm") < MIN_TEXT_CHARS
    assert result.pages[0].method == "skipped"


def test_empty_file_is_rejected():
    with pytest.raises(TextExtractionError, match="empty"):
        extract_text(_payload(b""))


def test_unopenable_pdf_is_rejected():
    with pytest.raises(TextExtractionError, match="Could not open PDF"):
        extract_text(_payload(b"%PDF-1.4 not really a pdf"))


def test_a_pdf_with_no_content_yields_empty_text_not_an_error():
    result = extract_text(_payload(_blank_pdf()))

    assert result.text == ""
    assert result.page_count == 1


def test_chars_reflects_the_extracted_text():
    result = extract_text(_payload(text_pdf(SAMPLE)))
    assert result.chars == len(result.text)


def test_skipped_pages_counts_only_skipped_ones():
    result = ExtractedText(
        text="x",
        page_count=3,
        pages=[PageText(0, "text", 1), PageText(1, "skipped", 0), PageText(2, "skipped", 0)],
    )
    assert result.skipped_pages == 2
