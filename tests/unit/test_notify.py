"""The filing notification is best-effort, and "best-effort" has to be provable.

Every test here is about the same property from a different angle: **nothing this module
does can reach the pipeline**. The document is filed and committed before it runs, so an
exception escaping would leave the SQS message undeleted, redeliver it, and re-run paid
stages on a document that is perfectly fine.
"""

import urllib.error

import pytest

from app.core.config import Settings
from app.services import notify

BASE = {"database_url": "postgresql+psycopg://u:p@h:5432/d"}

ARGS = {"document_id": 101, "section": "reports", "section_row_id": 412, "state": "classified"}


@pytest.fixture
def settings() -> Settings:
    return Settings(
        **BASE,
        spring_callback_url="http://spring.internal/internal/ai/filed",
        mhn_service_token="t" * 40,
    )


class _Response:
    status = 202

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_sends_the_document_and_where_it_went(settings, monkeypatch):
    sent = {}

    def fake_urlopen(request, timeout=None):
        sent["url"] = request.full_url
        sent["method"] = request.method
        sent["headers"] = request.headers
        sent["body"] = request.data
        sent["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    notify.document_filed(settings, **ARGS)

    assert sent["url"] == "http://spring.internal/internal/ai/filed"
    assert sent["method"] == "POST"
    assert sent["timeout"] == settings.spring_callback_timeout_seconds
    assert sent["headers"]["Authorization"] == f"Bearer {settings.mhn_service_token}"
    assert sent["body"] == (
        b'{"document_id": 101, "section": "reports", "section_row_id": 412, "state": "classified"}'
    )


def test_carries_no_user_and_no_document_content(settings, monkeypatch):
    """Spring resolves the owner from the row it already owns.

    Putting a user id on this payload would move an access decision out of the service that
    makes every other one — the same boundary the identity gate keeps. Anything read off
    the page has no business here at all.
    """
    body = {}

    def fake_urlopen(request, timeout=None):
        body["keys"] = set(__import__("json").loads(request.data).keys())
        return _Response()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    notify.document_filed(settings, **ARGS)

    assert body["keys"] == {"document_id", "section", "section_row_id", "state"}


def test_does_not_call_at_all_when_no_url_is_configured(monkeypatch):
    """An unset URL is the pre-feature behaviour exactly — not a call that fails quietly."""

    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("urlopen called with no callback URL configured")

    monkeypatch.setattr(notify.urllib.request, "urlopen", explode)

    notify.document_filed(Settings(**BASE, spring_callback_url=""), **ARGS)


@pytest.mark.parametrize(
    "boom",
    [
        urllib.error.HTTPError("http://x", 401, "Unauthorized", {}, None),
        urllib.error.HTTPError("http://x", 500, "Server Error", {}, None),
        urllib.error.URLError("connection refused"),
        TimeoutError("timed out"),
        ValueError("malformed"),
    ],
    ids=["401", "500", "unreachable", "timeout", "unexpected"],
)
def test_every_failure_is_swallowed(settings, monkeypatch, boom):
    """The pipeline must not be able to tell that this failed."""

    def fake_urlopen(request, timeout=None):
        raise boom

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    notify.document_filed(settings, **ARGS)  # no exception is the assertion


def test_a_rejection_logs_springs_own_status(settings, monkeypatch, caplog):
    """`type(exc).__name__` has hidden the difference between a 401 and a 404 in this
    integration before, and it cost hours each time. Keep the vendor's own words."""

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError("http://x", 403, "Forbidden", {}, None)

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level("WARNING"):
        notify.document_filed(settings, **ARGS)

    record = next(r for r in caplog.records if r.message == "filed_notify_rejected")
    assert record.status == 403
    assert record.reason == "Forbidden"


def test_settled_sends_only_the_document(settings, monkeypatch):
    """The settling announcement carries no section — that is what makes it usable for a
    name mismatch, which is rejected before filing and so lives in no section at all."""
    sent = {}

    def fake_urlopen(request, timeout=None):
        sent["url"] = request.full_url
        sent["body"] = request.data
        sent["headers"] = request.headers
        return _Response()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    notify.document_settled(settings, document_id=101)

    # Same endpoint as filing: one URL to configure, and Spring decides what to say by
    # reading its own tables rather than by being told.
    assert sent["url"] == "http://spring.internal/internal/ai/filed"
    assert sent["headers"]["Authorization"] == f"Bearer {settings.mhn_service_token}"
    assert sent["body"] == b'{"document_id": 101}'


def test_settled_never_raises(settings, monkeypatch):
    """Same contract as filing. This one runs after EVERY terminal outcome, including the
    failures, so an exception here would turn a handled rejection into a redelivery."""

    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("spring is down")

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    notify.document_settled(settings, document_id=101)


def test_settled_is_silent_without_a_callback_url(monkeypatch):
    called = False

    def fake_urlopen(request, timeout=None):  # pragma: no cover - must not run
        nonlocal called
        called = True
        return _Response()

    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)

    notify.document_settled(Settings(**BASE, mhn_service_token="t" * 40), document_id=101)

    assert called is False
