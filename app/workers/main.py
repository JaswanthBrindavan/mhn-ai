"""Report worker entrypoint.

Skeleton only. The SQS consume loop, bounded concurrency, and stage orchestration
land in step 5; this exists so the compose topology and image are testable now.
"""

import logging
import signal
import sys
import threading
from types import FrameType

from app.core.config import get_settings
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)

_shutdown = threading.Event()


def _handle_signal(signum: int, _frame: FrameType | None) -> None:
    logger.info("worker_shutdown_signal", extra={"signal": signum})
    _shutdown.set()


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info(
        "worker_started",
        extra={
            "max_concurrency": settings.worker_max_concurrency,
            "aws_mode": "localstack" if settings.uses_local_aws else "aws",
            "queue_configured": bool(settings.sqs_queue_url),
        },
    )

    if not settings.sqs_queue_url:
        logger.warning("worker_idle_no_queue_configured")

    # Placeholder: block until told to stop, so container lifecycle and graceful
    # shutdown can be exercised before the consume loop exists.
    _shutdown.wait()
    logger.info("worker_stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
