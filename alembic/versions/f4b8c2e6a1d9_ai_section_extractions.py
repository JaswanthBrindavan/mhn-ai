"""ai_section_extractions table

Revision ID: f4b8c2e6a1d9
Revises: e3c7a1f5b2d8
Create Date: 2026-07-27 16:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f4b8c2e6a1d9"
down_revision: str | None = "e3c7a1f5b2d8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ai_section_extractions",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("run_item_id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(length=32), nullable=False),
        sa.Column("data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("prompt_version", sa.String(length=32), nullable=False),
        sa.Column("schema_version", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_item_id"], ["ai_processing_run_items.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_item_id"),
    )
    op.create_index(
        "ix_ai_section_extractions_section", "ai_section_extractions", ["section"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_ai_section_extractions_section", table_name="ai_section_extractions")
    op.drop_table("ai_section_extractions")
