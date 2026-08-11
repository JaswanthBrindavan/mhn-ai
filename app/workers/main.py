"""Report worker: consume the queue and run reports through the pipeline.

Concurrency is bounded by ``WORKER_MAX_CONCURRENCY`` and enforced with a semaphore:
the loop only polls for as many messages as it has free slots, so it never holds more
in-flight work than it can process. Each message runs in its own thread with its own
database session (SQLAlchemy sessions are not thread-safe).

Scaling has two independent dials: threads per worker (this semaphore) and worker
replicas (``docker compose up --scale worker=N``). A burst is absorbed by the queue
and drained as slots free up.
"""

import logging
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import FrameType

from app.core.config import Settings, get_settings
from app.core.db import SessionLocal
from app.core.logging import configure_logging
from app.integrations.ai.factory import get_ai_provider
from app.integrations.aws import get_s3_client, get_sqs_client
from app.integrations.sqs import ReceivedMessage, receive_messages
from app.workers.processor import process_message
from app.workers.reaper import sweep_stale_items

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._shutdown = threading.Event()
        self._s3 = get_s3_client()
        self._sqs = get_sqs_client()
        self._ai = get_ai_provider()
        # One permit per concurrent slot. Acquired before a message is submitted,
        # released when it finishes — so the poll loop can size each receive to the
        # slots actually free.
        self._slots = threading.BoundedSemaphore(settings.worker_max_concurrency)
        self._pool = ThreadPoolExecutor(
            max_workers=settings.worker_max_concurrency, thread_name_prefix="report-worker"
        )
        # A third of the staleness window, so an item is picked up well before it has sat
        # for two full windows. Swept immediately on startup — a worker coming back after
        # a crash is exactly when there is something to recover.
        self._sweep_interval = max(30.0, settings.stale_item_timeout_seconds / 3)
        self._last_sweep = float("-inf")

    def request_shutdown(self, signum: int, _frame: FrameType | None) -> None:
        logger.info("worker_shutdown_signal", extra={"signal": signum})
        self._shutdown.set()

    def run(self) -> None:
        settings = self._settings
        logger.info(
            "worker_started",
            extra={
                "max_concurrency": settings.worker_max_concurrency,
                "aws_mode": "localstack" if settings.uses_local_aws else "aws",
                "queue_configured": bool(settings.sqs_queue_url),
            },
        )
        if not settings.sqs_queue_url:
            logger.error("worker_exit_no_queue_configured")
            return

        try:
            self._loop()
        finally:
            # Stop accepting work, then let in-flight items finish. Their messages
            # are only deleted on success, so anything cut short is redelivered.
            logger.info("worker_draining")
            self._pool.shutdown(wait=True)
            logger.info("worker_stopped")

    def _loop(self) -> None:
        while not self._shutdown.is_set():
            self._maybe_sweep()

            batch = self._acquire_slots()
            if batch == 0:
                continue  # shutdown requested while waiting for a slot

            try:
                messages = receive_messages(
                    self._sqs,
                    self._settings.sqs_queue_url,
                    max_messages=batch,
                    wait_seconds=self._settings.sqs_wait_time_seconds,
                    visibility_timeout=self._settings.sqs_visibility_timeout_seconds,
                )
            except Exception:
                # A receive failure (throttling, transient network) must not kill the
                # worker. Release the slots and back off via the next long poll.
                logger.exception("receive_failed")
                self._release(batch)
                continue

            # Release slots we reserved but did not fill.
            self._release(batch - len(messages))
            for message in messages:
                self._submit(message)

    def _maybe_sweep(self) -> None:
        """Run the stale-item sweep, at most once per interval.

        In the poll loop rather than a thread of its own: the loop already wakes every
        ``sqs_wait_time_seconds`` at worst, so a timestamp check is enough and there is no
        second thread to shut down cleanly. Every replica sweeps — ``FOR UPDATE SKIP
        LOCKED`` makes that safe, and it means recovery does not depend on one nominated
        worker being the one that is alive.
        """
        now = time.monotonic()
        if now - self._last_sweep < self._sweep_interval:
            return
        self._last_sweep = now

        session = SessionLocal()
        try:
            touched = sweep_stale_items(session, self._sqs, self._settings)
            if touched:
                logger.info("stale_sweep_touched_items", extra={"count": touched})
        except Exception:
            # A sweep failure must never take the worker down; it consumes no message and
            # the next tick tries again.
            logger.exception("stale_sweep_failed")
            session.rollback()
        finally:
            session.close()

    def _acquire_slots(self) -> int:
        """Block for one slot, then grab any others immediately available."""
        # Block (with a timeout so shutdown is responsive) for the first slot.
        while not self._slots.acquire(timeout=1):
            if self._shutdown.is_set():
                return 0
        count = 1
        while count < 10 and self._slots.acquire(blocking=False):
            count += 1
        return count

    def _release(self, n: int) -> None:
        for _ in range(n):
            self._slots.release()

    def _submit(self, message: ReceivedMessage) -> None:
        future = self._pool.submit(self._process, message)
        # Release the slot whatever happens, so a crash cannot leak capacity.
        future.add_done_callback(lambda _f: self._slots.release())

    def _process(self, message: ReceivedMessage) -> None:
        try:
            outcome = process_message(
                message,
                session_factory=SessionLocal,
                s3=self._s3,
                sqs=self._sqs,
                ai=self._ai,
                settings=self._settings,
            )
            logger.info(
                "message_processed",
                extra={"item_id": str(message.item_id), "outcome": outcome.value},
            )
        except Exception:
            # Never let one message take down a pool thread. The message was not
            # deleted, so it will be redelivered.
            logger.exception("message_processing_crashed", extra={"item_id": str(message.item_id)})


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)

    worker = Worker(settings)
    signal.signal(signal.SIGTERM, worker.request_shutdown)
    signal.signal(signal.SIGINT, worker.request_shutdown)
    worker.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
