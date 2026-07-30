"""The classifier is sent only the first N pages; the trim must be safe on any input."""

import io

import pypdfium2 as pdfium

from app.services.pdf_pages import limit_pdf_pages


def _pdf(n_pages: int) -> bytes:
    document = pdfium.PdfDocument.new()
    try:
        for _ in range(n_pages):
            document.new_page(200, 200)
        buffer = io.BytesIO()
        document.save(buffer)
    finally:
        document.close()
    return buffer.getvalue()


def _page_count(data: bytes) -> int:
    document = pdfium.PdfDocument(data)
    try:
        return len(document)
    finally:
        document.close()


def test_trims_multipage_pdf_to_the_limit():
    out = limit_pdf_pages(_pdf(5), 2)
    assert _page_count(out) == 2


def test_pdf_within_limit_is_returned_unchanged():
    data = _pdf(1)
    assert limit_pdf_pages(data, 2) is data


def test_non_pdf_bytes_returned_unchanged():
    data = b"\x89PNG\r\n\x1a\n" + b"not a pdf"
    assert limit_pdf_pages(data, 2) is data


def test_corrupt_pdf_falls_back_to_original():
    # Has the %PDF signature but is not parseable -> best-effort fallback, not a crash.
    data = b"%PDF-1.4 this is not really a pdf"
    assert limit_pdf_pages(data, 2) is data


def test_zero_max_pages_disables_trimming():
    data = _pdf(5)
    assert limit_pdf_pages(data, 0) is data
