# One image for both api and worker; they differ only by command.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Tesseract, the OCR binary. pytesseract is only a binding to it, so without this a
# document whose pages carry no text layer — a photographed or scanned insurance policy,
# scan report or vaccination card — fails with `text_extraction_failed`. Digital PDFs and
# every lab report are unaffected either way, which is what makes the omission easy to
# miss until a user uploads a photo.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so code edits do not invalidate the install layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini pyproject.toml ./
COPY alembic ./alembic
COPY app ./app

# Never run as root.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /srv
USER appuser

EXPOSE 8000

# Shell form so ${PORT} expands at runtime. Platforms that assign a port (Railway, Heroku,
# Cloud Run) inject it and route only to that port — a hardcoded 8000 there answers on a
# port nothing is forwarded to, which surfaces as "application failed to respond" with a
# perfectly healthy process in the logs. Defaults to 8000, so local and compose runs are
# unchanged.
#
# The WORKER runs from this same image with the command overridden to
# `python -m app.workers.main` — it is a second service, not a second container of this
# one. Without it documents are accepted and queued and then nothing processes them.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
