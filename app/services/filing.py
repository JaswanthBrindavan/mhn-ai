"""File a classified document into its section table.

Filing happens straight after classification, not at the end of the pipeline: a user who
uploaded into a section sees their document there in seconds rather than after the 30-80s
the AI stages take. Extraction and insights then UPDATE the row's ``content``.

**The ordering is the design.** S3 has no transactions, so the object copy cannot join the
database transaction. The order is:

1. copy the object (and its preview) into the section prefix
2. ONE transaction: INSERT the section row, record it on the run item, DELETE the intake row
3. after the commit, delete the original object

A crash after (1) leaves an orphan copy, which a redelivery overwrites. A crash after (2)
leaves an orphan original, which is cosmetic. What no crash can produce is a live row whose
``filepath`` points at a deleted object — which is exactly what deleting before the commit
would allow. Spring's own ``FileServiceImpl.moveUnclassified`` uses this order; ours matches
so that AI-filed and hand-filed documents are indistinguishable.
"""

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, Table, delete, insert, select, update
from sqlalchemy.orm import Session

from app.integrations.s3 import (
    SourceObjectMissingError,
    SourceObjectUnavailableError,
    copy_object,
    delete_object,
    object_exists,
)
from app.models.ai_results import AiSectionExtraction
from app.models.processing import AiProcessingRunItem
from app.models.spring import (
    insurance,
    reports,
    scans_imaging,
    unclassified_files,
    vaccinations,
)
from app.services.assembly import ContentState, build_content
from app.services.classification import DocumentSection
from app.services.s3_keys import key_for_section, preview_key_for
from app.workers.stagetypes import RejectStageError, TransientStageError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)

#: Section -> the Spring table a document of that section is filed into. A section absent
#: here cannot be filed; the router rejects it before filing is reached.
SECTION_TABLES: dict[DocumentSection, Table] = {
    DocumentSection.REPORTS: reports,
    DocumentSection.SCANS_IMAGING: scans_imaging,
    DocumentSection.INSURANCE: insurance,
    DocumentSection.VACCINATIONS: vaccinations,
}


def file_document(
    session: Session,
    s3: "S3Client",
    *,
    item_id: UUID,
    document_id: int,
    section: DocumentSection,
    content: dict[str, Any],
    bucket: str,
    expected: set[str],
) -> int | None:
    """Move the document into ``section`` and record it on the run item.

    Returns the section row's id, or None when there is nothing to file — either the run
    item no longer exists (its run was deleted; the message is stale), or the guard matched
    nothing because the item was cancelled or moved underneath us. In both cases nothing is
    filed and the intake row and its object are left intact. Already-filed items return their
    existing row id without doing any work, so a redelivered message is safe.
    """
    # FOR UPDATE, and it matters. Unlike the completion this replaced, the guarded UPDATE
    # below does not change ``status``, so it is not self-excluding: a second worker on the
    # same document would re-evaluate the same ``expected`` set, still match, and file the
    # document a second time — two rows in a Spring-owned table with one ``filepath``, one
    # of them orphaned and visible to the user. With the lock, the second worker blocks
    # here, then reads the first's committed ``section_row_id`` and takes the already-filed
    # return below, carrying on to extraction against the correct row.
    item = session.execute(
        select(AiProcessingRunItem.section_row_id)
        .where(AiProcessingRunItem.id == item_id)
        .with_for_update()
    ).one_or_none()
    if item is None:
        return None
    if item.section_row_id is not None:
        return int(item.section_row_id)

    src = session.execute(
        select(
            unclassified_files.c.user_id,
            unclassified_files.c.filepath,
            unclassified_files.c.private,
            unclassified_files.c.created_by,
        ).where(unclassified_files.c.id == document_id)
    ).one_or_none()
    if src is None:
        # Something else filed it (Spring's manual mover, or a lost race). Never fabricate
        # a section row: that would give the user the same document twice.
        session.rollback()
        logger.warning(
            "filing_source_missing", extra={"item_id": str(item_id), "document_id": document_id}
        )
        raise RejectStageError("source_document_missing", "Source document no longer exists")

    to_key = key_for_section(src.filepath, section.value)
    from_preview = preview_key_for(src.filepath)

    # Before the database, so a failure here leaves the original object and the intake row
    # untouched and the whole thing simply retries. The try wraps ONLY the S3 calls: widening
    # it would turn a programming error into a "transient" one and re-pay for classification
    # on every retry until the attempt cap.
    try:
        copy_object(s3, bucket, src.filepath, to_key)
        had_preview = object_exists(s3, bucket, from_preview)
        if had_preview:
            copy_object(s3, bucket, from_preview, preview_key_for(to_key))
    except SourceObjectMissingError as exc:
        # The realistic case is a user hand-filing the document in Spring between the SELECT
        # above and this copy. Permanent, not transient: retrying cannot bring it back.
        # The key stays out of the message — it is only ever internal detail.
        session.rollback()
        raise RejectStageError("source_object_missing", "Source file was not found") from exc
    except SourceObjectUnavailableError as exc:
        session.rollback()
        raise TransientStageError(f"source storage unavailable: {exc}") from exc

    table = SECTION_TABLES[section]
    row_id = session.execute(
        insert(table)
        .values(
            user_id=src.user_id,
            filepath=to_key,
            private=src.private,
            created_by=src.created_by,
            content=content,
        )
        .returning(table.c.id)
    ).scalar_one()

    filed = cast(
        "CursorResult[Any]",
        session.execute(
            update(AiProcessingRunItem)
            .where(
                AiProcessingRunItem.id == item_id,
                AiProcessingRunItem.status.in_(expected),
                # Defence in depth behind the row lock: this UPDATE never overwrites a
                # filing someone else recorded, whatever the status guard says.
                AiProcessingRunItem.section_row_id.is_(None),
            )
            .values(
                section_row_id=row_id,
                filed_section=section.value,
                # From here on, stages load the document from its new key.
                source_key=to_key,
            )
        ),
    ).rowcount
    if filed != 1:
        # Cancelled, moved, or filed underneath us: undo the insert entirely. The copied
        # object is left behind, harmless and overwritten if the document is filed later.
        session.rollback()
        return None

    session.execute(delete(unclassified_files).where(unclassified_files.c.id == document_id))
    session.commit()

    # Only now: the database points exclusively at the new keys.
    _delete_quietly(s3, bucket, src.filepath)
    if had_preview:
        _delete_quietly(s3, bucket, from_preview)

    logger.info(
        "document_filed",
        extra={"item_id": str(item_id), "section": section.value, "section_row_id": row_id},
    )
    return int(row_id)


def write_content(
    session: Session,
    item_id: UUID,
    content: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
) -> bool:
    """UPDATE a filed document's ``content``. False when the item was never filed.

    Not guarded on status: the row exists and its content should describe reality whatever
    the item's lifecycle state is. Cancellation is honoured by the caller, which decides
    *which* state to write.
    """
    item = session.execute(
        select(AiProcessingRunItem.section_row_id, AiProcessingRunItem.filed_section).where(
            AiProcessingRunItem.id == item_id
        )
    ).one_or_none()
    if item is None or item.section_row_id is None or item.filed_section is None:
        return False

    table = SECTION_TABLES[DocumentSection(item.filed_section)]
    session.execute(
        update(table)
        .where(table.c.id == item.section_row_id)
        .values(content=content, **(extra or {}))
    )
    session.commit()
    return True


def mark_content_failed(session: Session, item_id: UUID) -> None:
    """Stamp ``state: "failed"`` on a filed document that will not finish.

    Without this a cancelled or permanently-failed document keeps ``state: "classified"``
    and the app shows it as still processing for ever. A no-op for an unfiled document.
    """
    write_content(session, item_id, build_content(session, item_id, state=ContentState.FAILED))


def extra_columns(session: Session, item_id: UUID, section: DocumentSection) -> dict[str, Any]:
    """Section-table columns we can fill from the extraction, beyond ``content``.

    Only vaccinations has one today: ``next_due_on`` drives Spring's reminder index and we
    already extract the date. The rest (``hospital``, ``insurance.provider``) are foreign
    keys into master tables and need a name-to-id lookup, which is separate work.
    """
    if section is not DocumentSection.VACCINATIONS:
        return {}

    data = session.execute(
        select(AiSectionExtraction.data).where(AiSectionExtraction.run_item_id == item_id)
    ).scalar_one_or_none()
    # `or {}` rather than a get-default: a present-but-null "fields" would otherwise chain
    # off None and raise AttributeError, which is neither a reject nor a transient error.
    raw = ((data or {}).get("fields") or {}).get("next_due_date")
    if not raw:
        return {}
    try:
        # Already normalised to ISO by app.services.dates; parse rather than trust a shape.
        parsed = datetime.fromisoformat(str(raw))
    except ValueError:
        logger.warning("next_due_date_unparseable", extra={"item_id": str(item_id)})
        return {}
    return {"next_due_on": parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)}


def _delete_quietly(s3: "S3Client", bucket: str, key: str) -> None:
    """Remove a superseded object. A failure here is cosmetic — the row is already correct."""
    try:
        delete_object(s3, bucket, key)
    except SourceObjectUnavailableError as exc:
        logger.warning("filing_cleanup_failed", extra={"reason": str(exc)})
