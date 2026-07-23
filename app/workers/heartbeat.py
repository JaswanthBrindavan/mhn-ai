"""Keep a message invisible while its stage runs.

An AI stage can legitimately take minutes, longer than the queue's visibility
timeout. Without intervention SQS would redeliver the message to a second worker
mid-flight. This background heartbeat re-extends the visibility deadline on an
interval, so the message stays owned by the worker actually processing it.
"""

import logging
import threading
from typing import TYPE_CHECKING

from app.integrations.sqs import extend_visibility

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_sqs.client import SQSClient

logger = logging.getLogger(__name__)

#: Extend at least this often even for short timeouts, and never let the interval
#: reach the timeout itself — renew with comfortable margin before it lapses.
_MIN_INTERVAL_SECONDS = 15


class VisibilityHeartbeat:
    """Context manager that renews a message's visibility until the block exits."""

    def __init__(
        self,
        sqs: "SQSClient",
        queue_url: str,
        receipt_handle: str,
        visibility_timeout: int,
    ) -> None:
        self._sqs = sqs
        self._queue_url = queue_url
        self._receipt_handle = receipt_handle
        self._visibility_timeout = visibility_timeout
        # Renew at a third of the timeout so a single missed beat is not fatal.
        self._interval = max(_MIN_INTERVAL_SECONDS, visibility_timeout // 3)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "VisibilityHeartbeat":
        self._thread = threading.Thread(target=self._run, name="visibility-heartbeat", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        # wait() returns True when stopped; loop body runs only on timeout tick.
        while not self._stop.wait(self._interval):
            ok = extend_visibility(
                self._sqs, self._queue_url, self._receipt_handle, self._visibility_timeout
            )
            if not ok:
                # The receipt is gone or the message already moved on; nothing this
                # thread can do, so stop rather than spin on the same error.
                break

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
