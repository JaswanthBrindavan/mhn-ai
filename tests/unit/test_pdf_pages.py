"""The classifier is sent a few pages; the trim must be safe on any input.

Since 2026-09-05 those pages are the FRONT ones plus one sampled from deeper in, so the
tests here pin two things the old first-N version could not get wrong: that a page past
the front actually reaches the classifier, and that the choice is stable for a given
document. Both are load-bearing — see `pdf_pages._sample_index`.

Each page is built a distinct WIDTH (100 + its index), so the output identifies exactly
which source pages survived. Comparing serialised bytes does not work: PDFium stamps a
fresh document ID into every save, so two identical selections produce different files.
"""

import io

import pypdfium2 as pdfium

from app.services.pdf_pages import _sample_index, limit_pdf_pages


def _pdf(n_pages: int) -> bytes:
    """``n_pages`` blank pages, page *i* being ``100 + i`` wide so it names itself."""
    document = pdfium.PdfDocument.new()
    try:
        for index in range(n_pages):
            document.new_page(100 + index, 200)
        buffer = io.BytesIO()
        document.save(buffer)
    finally:
        document.close()
    return buffer.getvalue()


def _kept(data: bytes) -> list[int]:
    """The source page indexes present in ``data``, recovered from their widths."""
    document = pdfium.PdfDocument(data)
    try:
        return [round(document[i].get_width()) - 100 for i in range(len(document))]
    finally:
        document.close()


def _page_count(data: bytes) -> int:
    document = pdfium.PdfDocument(data)
    try:
        return len(document)
    finally:
        document.close()


def test_trims_multipage_pdf_to_the_limit():
    assert _page_count(limit_pdf_pages(_pdf(5), 2)) == 2


def test_a_page_past_the_front_is_included():
    """The whole point of the 2026-09-05 change: the classifier must see content, not only
    front matter. A case-sheet cover and an EMPTY table of contents were all it got for a
    43-page investigation bundle, which it then classified `medical_condition` and
    rejected unread."""
    kept = _kept(limit_pdf_pages(_pdf(20), 3))

    assert kept[:2] == [0, 1]  # the front pages, in order
    assert kept[2] >= 2  # and one from deeper in


def test_the_sampled_page_is_stable_for_the_same_document():
    """A per-call random page would let one PDF classify as `reports` on one attempt and
    `scans_imaging` on the next — a retry of a rejected document really does re-read it
    (`processor._reading_already_made` excludes routing rejections on purpose)."""
    data = _pdf(30)

    assert _kept(limit_pdf_pages(data, 3)) == _kept(limit_pdf_pages(data, 3))


def test_different_documents_sample_different_pages():
    """Otherwise this is a hardcoded third page wearing a hash."""
    picks = {_sample_index(f"doc-{n}".encode(), 2, 40) for n in range(60)}

    assert len(picks) > 10


def test_the_sampled_index_stays_inside_the_document():
    """An index past the last page would raise inside `import_pages`, and the best-effort
    handler would swallow it — silently sending the classifier the untrimmed file."""
    for pages in range(3, 60):
        assert 2 <= _sample_index(f"x{pages}".encode(), 2, pages - 1) <= pages - 1


def test_a_single_page_limit_keeps_the_first_page():
    """No room for front matter AND a sample; the front page is the more useful one."""
    assert _kept(limit_pdf_pages(_pdf(9), 1)) == [0]


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
