"""Tell Spring, the moment a document becomes visible in a section.

The app polls ``ai-status`` every few seconds and learns the same thing, so this exists
only to close that gap: with it, the screen can move the user to the section as the row
appears rather than up to a poll interval later.

**Three properties, and each one is load-bearing.**

*Best-effort.* Nothing here may raise into the pipeline. The document is already filed and
committed by the time this runs; a failed announcement that propagated would leave the SQS
message undeleted, redeliver it, and re-run every paid stage on a document that is
perfectly fine. The poll is the backstop, which is precisely why no retry, backoff or
outbound dead-letter queue is needed.

*Post-commit.* Called only after ``filing.file_document`` has committed. Announcing a row
Spring cannot yet read is the same bug as publishing to SQS before the commit.

*No user, no content.* The payload says which document went where — never who owns it and
never anything off the page. Spring resolves the owner from the section row it already
owns, so the decision about **who may be told** stays on the side that makes every other
access decision. This is the same boundary the identity gate keeps.
"""

import json
import logging
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.core.config import Settings

logger = logging.getLogger(__name__)


def document_filed(
    settings: "Settings",
    *,
    document_id: int,
    section: str,
    section_row_id: int,
    state: str,
) -> None:
    """Announce that ``document_id`` now lives in ``section`` as ``section_row_id``.

    ``state`` is the ``content.ai.state`` the row was filed with — ``classified`` when the
    document is filed and waiting to be analysed, which is what the app needs to know to
    decide whether to offer the Analyse button on arrival.

    Never raises. A disabled URL, an unreachable Spring, a 500, a timeout and a malformed
    response are all one thing here: the app finds out by polling instead.
    """
    _announce(
        settings,
        document_id=document_id,
        payload={
            "document_id": document_id,
            "section": section,
            "section_row_id": section_row_id,
            "state": state,
        },
    )


def document_settled(settings: "Settings", *, document_id: int) -> None:
    """Announce that ``document_id``'s run has come to rest.

    Distinct from :func:`document_filed`, and not a replacement for it, because they are
    different moments. Filing is when the row appears and the screen can move to it — it
    happens BEFORE the run item completes, and in the analyse-now path a further minute of
    stages follows. Settling is when nothing more will happen without the reader.

    That second moment is the one worth a notification, and it is the only one that can
    describe **a document that was never filed at all**: a name mismatch is rejected before
    filing, so it lives in no section, and a reader who opens the app cannot find it by
    looking. Hence no ``section`` here — Spring reads what the document is waiting on out of
    its own tables, so this service never has to know what a notification is.

    Never raises, for the same reason as its sibling: the sweep behind it is the backstop.
    """
    _announce(settings, document_id=document_id, payload={"document_id": document_id})


def _announce(settings: "Settings", *, document_id: int, payload: dict) -> None:
    """POST one announcement to Spring, swallowing everything. See the module docstring."""
    if not settings.spring_callback_url:
        return

    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        settings.spring_callback_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.mhn_service_token}",
        },
    )
    try:
        with urllib.request.urlopen(
            request, timeout=settings.spring_callback_timeout_seconds
        ) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        # Keep Spring's own words: a 401 here and a 404 here need different fixes, and
        # `type(exc).__name__` has hidden the difference in this integration before.
        logger.warning(
            "filed_notify_rejected",
            extra={"document_id": document_id, "status": exc.code, "reason": exc.reason},
        )
        return
    except Exception as exc:
        # Deliberately total. Anything at all that goes wrong here is one thing to the
        # pipeline: the app finds out by polling instead. See the module docstring.
        logger.warning(
            "filed_notify_failed",
            extra={"document_id": document_id, "error": f"{type(exc).__name__}: {exc}"},
        )
        return

    logger.info(
        "filed_notify_sent",
        # `section` is absent on a settling announcement, which is the whole point of it:
        # a name mismatch has no section, and that is the case most worth telling about.
        extra={"document_id": document_id, "section": payload.get("section"), "status": status},
    )
