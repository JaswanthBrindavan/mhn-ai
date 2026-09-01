"""Build real PDFs for tests that read a document rather than just pass bytes around.

Most fixtures get away with placeholder bytes (`b"%PDF-1.4 fake report"`), because the
report pipeline hands the file to the model without opening it. Anything that goes through
``app.services.text_layer`` does open it, so those tests need a genuine PDF.

reportlab is a dev-only dependency: neither pdfplumber nor pypdfium2 can *create* a PDF —
they read and render — and a fixture written in code beats a committed binary.
"""

import io

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

#: A left-hand label and a right-hand value, which is the shape of every report header and
#: the case position-sorted text extraction exists to get right.
TWO_COLUMN_GAP = 300


def text_pdf(body: str, *, pages: int = 1, line_height: int = 14) -> bytes:
    """A PDF with a real text layer, one line of ``body`` per line, repeated per page."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    for _ in range(pages):
        cursor = A4[1] - 72
        # `or [""]` keeps an empty body a real (blank) page rather than no page at all.
        for line in body.splitlines() or [""]:
            pdf.drawString(72, cursor, line)
            cursor -= line_height
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def two_column_pdf(rows: list[tuple[str, str]]) -> bytes:
    """A single page of label/value pairs, the label left and the value far right."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    cursor = A4[1] - 72
    for label, value in rows:
        pdf.drawString(72, cursor, label)
        pdf.drawString(TWO_COLUMN_GAP, cursor, value)
        cursor -= 20
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()
