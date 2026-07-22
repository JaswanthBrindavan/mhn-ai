"""Service-owned models.

Importing this package registers every ``ai_*`` table on ``Base.metadata``. Alembic's
``env.py`` imports it for exactly that reason — a model not imported here is invisible
to autogenerate and will silently never get a migration.

``app.models.spring`` is deliberately NOT re-exported: those tables live on a separate
MetaData so they can never become migration targets.
"""

from app.models.enums import (
    ACTIVE_STATUSES,
    CANCELLABLE_STATUSES,
    TERMINAL_STATUSES,
    RunItemStatus,
)
from app.models.processing import AiProcessingRun, AiProcessingRunItem

__all__ = [
    "ACTIVE_STATUSES",
    "CANCELLABLE_STATUSES",
    "TERMINAL_STATUSES",
    "AiProcessingRun",
    "AiProcessingRunItem",
    "RunItemStatus",
]
