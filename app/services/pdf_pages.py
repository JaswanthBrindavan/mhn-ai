"""Trim a PDF to a few pages, for cheaper classification.

Classification only needs to recognise the document type — so the classifier is sent a
trimmed PDF while extraction still reads the whole document. This is a best-effort cost
optimisation: anything that is not a parseable PDF, or fails to trim, is returned
unchanged so classification never breaks over it.

**The trim is the FRONT pages plus one page from deeper in, not simply the first N**
(2026-09-05). Taking only the front assumes a document announces itself on page one, and
a hospital bundle does not: document 214 was 43 pages of investigations whose first two
pages were a case-sheet cover and an EMPTY table of contents. It classified
``medical_condition`` at 0.9 confidence — correct for what it was shown — was rejected as
unfilable, and was never read, while page 5 held a blood count with a WBC of 22,640
against a range of 5,000-15,000.

The sampled page is chosen from the document's own bytes rather than from ``random``. See
``_sample_index``: a per-call random page would make the same document classify
differently on different passes, which is the one property this stage cannot lose.

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
is still present. That is why ``text_layer.py`` reads the text layer through pdfplumber,
which sorts by position, rather than through pypdfium2. (This paragraph named ``ocr.py``
until 2026-09-05; that module was deleted with OCR on 2026-09-01 and nothing rasterises a
page any more.) Trimming is unaffected either way: it copies whole pages and never reads
their text.
"""

import hashlib
import io
import logging

import pypdfium2 as pdfium

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF"


def _sample_index(data: bytes, low: int, high: int) -> int:
    """A page index in ``[low, high]``, picked arbitrarily but STABLY for a document.

    Derived from the document's own bytes, not from ``random``, and the difference is not
    cosmetic. Every stage runs at ``temperature=0`` so that a second pass over the same
    document reaches the same section, and the retry path for a ``rejected`` document
    genuinely re-reads it (``processor._reading_already_made`` deliberately excludes
    routing rejections). A per-call random page would therefore let one PDF classify as
    ``reports`` on one attempt and ``scans_imaging`` on the next — this bundle has pages
    that say both — turning retry into a dice roll and reintroducing exactly the
    disagreement ``classification.adopt_prior`` exists to make impossible.

    It would also make the stored ``reasoning`` undebuggable: it would describe a page
    nobody could identify afterwards. Hence ``pdf_trimmed`` logs the index it chose.

    Different documents still land on different pages, which is the point of not
    hardcoding one position.
    """
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return low + int.from_bytes(digest, "big") % (high - low + 1)


def limit_pdf_pages(data: bytes, max_pages: int) -> bytes:
    """Return ``data`` reduced to ``max_pages`` pages, or unchanged.

    The kept pages are the first ``max_pages - 1`` plus one sampled from the rest, so the
    classifier sees both the letterhead that names the document and a page of its actual
    content. ``max_pages == 1`` keeps the first page alone — there is no room for both.

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

        # pages_before > max_pages is guaranteed above, so the sample range is never empty.
        keep = list(range(max_pages - 1)) if max_pages >= 2 else [0]
        if max_pages >= 2:
            keep.append(_sample_index(data, max_pages - 1, pages_before - 1))

        trimmed = pdfium.PdfDocument.new()
        trimmed.import_pages(source, keep)
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

    # `kept` names the actual page indexes, not just how many: which page was sampled is
    # the whole diagnosis when a classification looks wrong, and it is the one thing a
    # reader of the stored `reasoning` cannot otherwise recover.
    logger.info(
        "pdf_trimmed",
        extra={"pages_before": pages_before, "pages_after": len(keep), "kept": keep},
    )
    return buffer.getvalue()
