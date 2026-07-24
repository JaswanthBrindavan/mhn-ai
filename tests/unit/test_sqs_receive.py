"""Message parsing and the visibility heartbeat, with a fake SQS client."""

import json
import threading
import uuid

from app.integrations.sqs import MESSAGE_SCHEMA_VERSION, _parse
from app.workers.heartbeat import VisibilityHeartbeat


def _raw(body: dict, receipt: str = "rh", receive_count: int = 1) -> dict:
    return {
        "MessageId": "m1",
        "ReceiptHandle": receipt,
        "Body": json.dumps(body),
        "Attributes": {"ApproximateReceiveCount": str(receive_count)},
    }


def _valid_body() -> dict:
    return {
        "schema_version": MESSAGE_SCHEMA_VERSION,
        "item_id": str(uuid.uuid4()),
        "run_id": str(uuid.uuid4()),
        "document_id": 7,
        "attempt": 0,
    }


# --- parsing ----------------------------------------------------------------


def test_valid_message_parses():
    body = _valid_body()
    msg = _parse(_raw(body, receipt="handle-1", receive_count=3))
    assert msg is not None
    assert str(msg.item_id) == body["item_id"]
    assert msg.document_id == 7
    assert msg.receipt_handle == "handle-1"
    assert msg.approx_receive_count == 3


def test_unparseable_body_is_dropped():
    assert _parse({"ReceiptHandle": "x", "Body": "{not json"}) is None


def test_unknown_schema_version_is_dropped():
    body = _valid_body() | {"schema_version": 999}
    assert _parse(_raw(body)) is None


def test_missing_field_is_dropped():
    body = _valid_body()
    del body["item_id"]
    assert _parse(_raw(body)) is None


def test_non_integer_report_id_is_dropped():
    body = _valid_body() | {"document_id": "not-a-number"}
    assert _parse(_raw(body)) is None


# --- heartbeat --------------------------------------------------------------


class _FakeSqs:
    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def change_message_visibility(self, **_kwargs):
        with self._lock:
            self.calls += 1


def test_heartbeat_extends_visibility_on_its_interval(monkeypatch):
    # Force a tiny interval so the test does not wait.
    monkeypatch.setattr("app.workers.heartbeat._MIN_INTERVAL_SECONDS", 0)
    fake = _FakeSqs()

    beat = VisibilityHeartbeat(fake, "q", "rh", visibility_timeout=0)
    # Interval becomes max(0, 0//3) = 0, so it renews rapidly.
    beat._interval = 0.02
    with beat:
        _wait_until(lambda: fake.calls >= 2, timeout=2.0)

    assert fake.calls >= 2


def test_heartbeat_stops_after_block_exits(monkeypatch):
    fake = _FakeSqs()
    beat = VisibilityHeartbeat(fake, "q", "rh", visibility_timeout=0)
    beat._interval = 0.02
    with beat:
        _wait_until(lambda: fake.calls >= 1, timeout=2.0)
    stopped_at = fake.calls
    _sleep(0.1)
    # No further renewals once the block has exited.
    assert fake.calls == stopped_at


def _wait_until(cond, timeout: float) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.01)


def _sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)
