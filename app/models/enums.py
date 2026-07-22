"""Processing lifecycle states.

    pending -> queued -> processing -> classifying -> extracting
            -> generating_insights -> completed

Terminal alternatives: failed, rejected, cancelled.

Stored as VARCHAR with a CHECK constraint rather than a native PostgreSQL enum.
Adding a stage later is then an ordinary migration, instead of ``ALTER TYPE ... ADD
VALUE`` with its transaction-block restrictions and awkward downgrades.
"""

from enum import StrEnum


class RunItemStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    CLASSIFYING = "classifying"
    EXTRACTING = "extracting"
    GENERATING_INSIGHTS = "generating_insights"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


#: Work is finished; nothing further will happen without an explicit retry.
TERMINAL_STATUSES: frozenset[RunItemStatus] = frozenset(
    {
        RunItemStatus.COMPLETED,
        RunItemStatus.FAILED,
        RunItemStatus.REJECTED,
        RunItemStatus.CANCELLED,
    }
)

#: An item in one of these states is in flight. The partial unique index on
#: ``report_id`` uses exactly this set, so a report can never have two active items.
ACTIVE_STATUSES: frozenset[RunItemStatus] = frozenset(RunItemStatus) - TERMINAL_STATUSES

#: Statuses a cancellation request may move an item out of.
CANCELLABLE_STATUSES: frozenset[RunItemStatus] = ACTIVE_STATUSES
