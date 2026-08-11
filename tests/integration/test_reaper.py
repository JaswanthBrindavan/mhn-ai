"""The stale-item sweep: recovery for work that stopped with no message in flight.

The case that matters here is an item stuck at ``queued``. Local and production share one
SQS queue, so whichever worker wins a receive looks the item up in *its own* database,
does not find it, and deletes the message. Our item was never claimed, so it never reaches
an in-progress state — and because it was never classified the document is never filed, so
it sits in ``unclassified_files`` and the app shows it pending under the section the user
chose, for ever, with nothing generated.

The backlog entry this was built from swept the in-progress states only and would have
missed that entirely. It also claimed the heartbeat keeps ``updated_at`` fresh, which it
does not — the heartbeat extends SQS visibility and touches no row. Both corrected in
``docs/FUTURE.md`` before any of this was written.
"""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update

from app.models.enums import RunItemStatus
from app.models.processing import AiProcessingRunItem
from app.workers.reaper import sweep_stale_items


def _age(db_session, item_id: uuid.UUID, seconds: int) -> None:
    """Backdate an item's ``updated_at`` past the staleness window.

    Set explicitly rather than by waiting, and with ``synchronize_session=False`` so the
    ORM's ``onupdate`` does not immediately stamp it back to now.
    """
    db_session.execute(
        update(AiProcessingRunItem)
        .where(AiProcessingRunItem.id == item_id)
        .values(updated_at=datetime.now(UTC) - timedelta(seconds=seconds))
        .execution_options(synchronize_session=False)
    )
    db_session.commit()


def _item(db_session, api, make_document) -> tuple[uuid.UUID, int]:
    document_id = make_document()
    run = api.post(
        "/v1/document-processing-runs", json={"documents": [{"document_id": document_id}]}
    ).json()
    return uuid.UUID(run["items"][0]["item_id"]), document_id


def _status(db_session, item_id: uuid.UUID) -> tuple[str, int]:
    row = db_session.execute(
        select(AiProcessingRunItem.status, AiProcessingRunItem.attempt_count).where(
            AiProcessingRunItem.id == item_id
        )
    ).one()
    return row.status, row.attempt_count


def _queue_depth(sqs, queue_url) -> int:
    attributes = sqs.get_queue_attributes(
        QueueUrl=queue_url, AttributeNames=["ApproximateNumberOfMessages"]
    )["Attributes"]
    return int(attributes["ApproximateNumberOfMessages"])


def test_a_queued_item_whose_message_vanished_is_republished(
    api, db_session, make_document, aws, test_settings
):
    """The shared-queue case, and the whole reason `queued` is in the sweep."""
    _, sqs, queue_url, _ = aws
    item_id, _ = _item(db_session, api, make_document)
    # Exactly what the other environment's worker does: take the message and drop it.
    sqs.purge_queue(QueueUrl=queue_url)
    assert _queue_depth(sqs, queue_url) == 0
    assert _status(db_session, item_id) == (RunItemStatus.QUEUED.value, 0)

    _age(db_session, item_id, test_settings.stale_item_timeout_seconds + 60)
    touched = sweep_stale_items(db_session, sqs, test_settings)

    assert touched == 1
    assert _queue_depth(sqs, queue_url) == 1
    # An attempt was spent, which is what bounds this. See the exhaustion test below.
    assert _status(db_session, item_id) == (RunItemStatus.QUEUED.value, 1)


def test_a_fresh_item_is_left_alone(api, db_session, make_document, aws, test_settings):
    """The guard is ``updated_at``, and a live worker keeps it moving via stage
    transitions. Sweeping an item somebody is working on would pay for the AI twice."""
    _, sqs, queue_url, _ = aws
    item_id, _ = _item(db_session, api, make_document)
    sqs.purge_queue(QueueUrl=queue_url)

    # Not backdated: it has only just been queued.
    assert sweep_stale_items(db_session, sqs, test_settings) == 0
    assert _queue_depth(sqs, queue_url) == 0
    assert _status(db_session, item_id) == (RunItemStatus.QUEUED.value, 0)


def test_the_requeue_is_bounded_and_ends_failed(api, db_session, make_document, aws, test_settings):
    """The trap this had to be designed around.

    An item stuck at ``queued`` has ``attempt_count = 0`` and would keep it: the counter
    only moves in ``claim_item``, which never runs for a message that was eaten. So
    ``MAX_ATTEMPTS`` does not bound the loop on its own, and a naive sweep re-publishes
    the same document every interval for ever — back into the same race that ate it.
    The reaper spends an attempt itself, so the existing cap applies.
    """
    _, sqs, queue_url, _ = aws
    item_id, _ = _item(db_session, api, make_document)
    sqs.purge_queue(QueueUrl=queue_url)

    for expected_attempt in range(1, test_settings.max_attempts + 1):
        _age(db_session, item_id, test_settings.stale_item_timeout_seconds + 60)
        sweep_stale_items(db_session, sqs, test_settings)
        assert _status(db_session, item_id) == (RunItemStatus.QUEUED.value, expected_attempt)
        sqs.purge_queue(QueueUrl=queue_url)  # eaten again, every time

    # Cap reached: the next sweep gives up rather than publishing a fourth time.
    _age(db_session, item_id, test_settings.stale_item_timeout_seconds + 60)
    sweep_stale_items(db_session, sqs, test_settings)

    status, _ = _status(db_session, item_id)
    assert status == RunItemStatus.FAILED.value
    assert _queue_depth(sqs, queue_url) == 0
    code = db_session.execute(
        select(AiProcessingRunItem.last_error_code).where(AiProcessingRunItem.id == item_id)
    ).scalar_one()
    assert code == "stale_item_abandoned"


def test_a_terminal_item_is_never_swept(api, db_session, make_document, aws, test_settings):
    """Completed, failed, rejected and cancelled are done. Re-queueing a completed
    document would re-run paid stages over a result that is already final."""
    _, sqs, queue_url, _ = aws
    item_id, _ = _item(db_session, api, make_document)
    db_session.execute(
        update(AiProcessingRunItem)
        .where(AiProcessingRunItem.id == item_id)
        .values(status=RunItemStatus.COMPLETED.value)
    )
    db_session.commit()
    sqs.purge_queue(QueueUrl=queue_url)
    _age(db_session, item_id, test_settings.stale_item_timeout_seconds + 60)

    assert sweep_stale_items(db_session, sqs, test_settings) == 0
    assert _queue_depth(sqs, queue_url) == 0


def test_an_item_stuck_mid_pipeline_is_also_recovered(
    api, db_session, make_document, aws, test_settings
):
    """The case the original backlog entry described: a worker died mid-stage and the
    message was never redelivered."""
    _, sqs, queue_url, _ = aws
    item_id, _ = _item(db_session, api, make_document)
    db_session.execute(
        update(AiProcessingRunItem)
        .where(AiProcessingRunItem.id == item_id)
        .values(status=RunItemStatus.EXTRACTING.value)
    )
    db_session.commit()
    sqs.purge_queue(QueueUrl=queue_url)
    _age(db_session, item_id, test_settings.stale_item_timeout_seconds + 60)

    assert sweep_stale_items(db_session, sqs, test_settings) == 1
    assert _queue_depth(sqs, queue_url) == 1
    # Back to queued, so claim_item picks it up from the top like any redelivery.
    assert _status(db_session, item_id) == (RunItemStatus.QUEUED.value, 1)
