"""auto-filing columns on ai_processing_run_items

Revision ID: b6d1f8a3c209
Revises: a7d2e9c4b3f6
Create Date: 2026-08-01 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b6d1f8a3c209"
down_revision: str | None = "a7d2e9c4b3f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The filed row is no longer always a report.
    op.alter_column(
        "ai_processing_run_items", "reports_id", new_column_name="section_row_id"
    )
    op.add_column(
        "ai_processing_run_items", sa.Column("filed_section", sa.String(32), nullable=True)
    )
    op.add_column(
        "ai_processing_run_items", sa.Column("intended_section", sa.String(32), nullable=True)
    )
    op.add_column(
        "ai_processing_run_items", sa.Column("source_key", sa.String(500), nullable=True)
    )

    # Backfill source_key for items created before it existed, so no code path needs a
    # fallback to unclassified_files. This reads a Spring-owned table but alters only ours —
    # it is a data backfill, not DDL against their schema.
    #
    # Skipped when that table is absent. On a database Spring has not migrated yet the
    # backfill has nothing to do anyway — no intake rows means no items referencing them —
    # so failing here would only make this service's migrations depend on another service
    # having deployed first, which is not a dependency this service accepts anywhere else.
    intake_exists = (
        op.get_bind()
        .execute(sa.text("SELECT to_regclass('public.unclassified_files')"))
        .scalar()
        is not None
    )
    if intake_exists:
        op.execute(
            """
            UPDATE ai_processing_run_items AS i
               SET source_key = u.filepath
              FROM unclassified_files AS u
             WHERE u.id = i.document_id
               AND i.source_key IS NULL
            """
        )
    # Items already filed under the old behaviour were reports by definition.
    op.execute(
        """
        UPDATE ai_processing_run_items
           SET filed_section = 'reports'
         WHERE section_row_id IS NOT NULL AND filed_section IS NULL
        """
    )


def downgrade() -> None:
    op.drop_column("ai_processing_run_items", "source_key")
    op.drop_column("ai_processing_run_items", "intended_section")
    op.drop_column("ai_processing_run_items", "filed_section")
    op.alter_column(
        "ai_processing_run_items", "section_row_id", new_column_name="reports_id"
    )
