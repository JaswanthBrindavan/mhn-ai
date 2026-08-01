"""Load the source document for a run item, shared by every AI stage.

Classification and extraction both need the same object from S3 with the same
reject/transient semantics, so the logic lives here once. A missing key/object or an
unsupported type is a permanent reject; a storage outage is transient (retry later).

The key comes from the run item (``StageContext.source_key``), never from the intake
table: filing deletes that row mid-pipeline, so a stage that looked the document up
there would reject every post-filing document as missing. This module therefore touches
no Spring-owned table at all.
"""

from app.integrations.ai.base import DocumentPayload
from app.integrations.s3 import (
    SourceObjectMissingError,
    SourceObjectUnavailableError,
    get_object,
)
from app.services.source_validation import resolve_content_type
from app.workers.stagetypes import RejectStageError, StageContext, TransientStageError


def load_source_document(ctx: StageContext) -> DocumentPayload:
    """Fetch and validate the item's source object, ready to hand to the model."""
    if not ctx.source_key:
        # Set at submit for every item and updated at filing, so a gap is an anomaly
        # rather than a retryable condition.
        raise RejectStageError("source_document_missing", "Source document no longer exists")

    try:
        content = get_object(ctx.s3, ctx.settings.s3_bucket, ctx.source_key)
    except SourceObjectMissingError as exc:
        raise RejectStageError("source_object_missing", "Source file was not found") from exc
    except SourceObjectUnavailableError as exc:
        raise TransientStageError(f"source storage unavailable: {exc}") from exc

    content_type = resolve_content_type(content.metadata)
    if content_type is None or content_type not in ctx.settings.allowed_content_type_set:
        # Validated at submit, but the object could have changed underneath us.
        raise RejectStageError("unsupported_content_type", "Source file type is not supported")

    return DocumentPayload(data=content.data, content_type=content_type, filename=ctx.source_key)
