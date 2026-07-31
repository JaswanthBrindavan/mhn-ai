"""Trim a PDF to its first N pages, for cheaper classification.

Classification only needs to recognise the document type, which the first page or two
reveal — so the classifier is sent a trimmed PDF while extraction still reads the whole
document. This is a best-effort cost optimisation: anything that is not a parseable PDF,
or fails to trim, is returned unchanged so classification never breaks over it.

Detection is by file signature (the ``%PDF`` magic), not the declared MIME type, so a
mislabelled upload is still handled correctly. Logs carry page counts and error types
only — never document content.

Built on ``pypdfium2`` (PDFium, BSD-3/Apache-2.0), measured against pypdf, PyMuPDF and
pdfplumber on the sample reports: fastest of the four and the only permissively-licensed
one that can also rasterise pages. See ``docs/extraction-cost-options.md``.

That benchmark also judged its text column-order-preserving. A later field-level
measurement disagreed. pypdfium2 always returns *storage* order — the order the
generating software happened to emit characters in — which usually matches reading order
and sometimes does not; when it does not, the failure is silent, because every character
is still present. That is why ``ocr.py`` reads the text layer through pdfplumber, which
sorts by position, rather than through pypdfium2 — though pypdfium2 still opens the
document and rasterises pages for OCR there. Trimming is unaffected either way: it copies
whole pages and never reads their text.
"""

import io
import logging

import pypdfium2 as pdfium

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF"


def limit_pdf_pages(data: bytes, max_pages: int) -> bytes:
    """Return ``data`` reduced to its first ``max_pages`` pages, or unchanged.

    Unchanged when: trimming is disabled (``max_pages < 1``), the bytes are not a PDF,
    the PDF already has ``<= max_pages`` pages, or parsing/writing fails.
    """
    if max_pages < 1:
        logger.debug("pdf_trim_disabled")
        return data
    if not data.startswith(_PDF_MAGIC):
        logger.debug("pdf_trim_skipped_not_pdf")
        return data

    source = trimmed = None
    try:
        source = pdfium.PdfDocument(data)
        pages_before = len(source)
        if pages_before <= max_pages:
            logger.debug("pdf_trim_skipped_within_limit", extra={"pages": pages_before})
            return data

        trimmed = pdfium.PdfDocument.new()
        trimmed.import_pages(source, list(range(max_pages)))
        buffer = io.BytesIO()
        trimmed.save(buffer)
    except Exception as exc:
        # Best effort: a trim failure forfeits the saving for this document, never the
        # classification. Log the exception type only (no content).
        logger.warning("pdf_trim_failed", extra={"error_type": type(exc).__name__})
        return data
    finally:
        # PDFium holds native handles; these are not garbage-collected for us.
        for document in (trimmed, source):
            if document is not None:
                document.close()

    logger.info("pdf_trimmed", extra={"pages_before": pages_before, "pages_after": max_pages})
    return buffer.getvalue()
