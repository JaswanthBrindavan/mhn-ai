"""ai_thp_fallbacks table (R&D worklist)

Revision ID: e3c7a1f5b2d8
Revises: c2a5b9d3e7f1
Create Date: 2026-07-27 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e3c7a1f5b2d8"
down_revision: str | None = "c2a5b9d3e7f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ai_thp_fallbacks",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("run_item_id", sa.UUID(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("test_name", sa.String(length=256), nullable=False),
        sa.Column("matched_parameter", sa.String(length=256), nullable=True),
        sa.Column("group_attempted", sa.String(length=64), nullable=True),
        sa.Column("reason", sa.String(length=32), nullable=False),
        sa.Column("patient_age", sa.String(length=32), nullable=True),
        sa.Column("patient_gender", sa.String(length=32), nullable=True),
        sa.Column("report_reference_range", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_item_id"], ["ai_processing_run_items.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ai_thp_fallbacks_reason", "ai_thp_fallbacks", ["reason"])
    op.create_index(
        "ix_ai_thp_fallbacks_matched_parameter", "ai_thp_fallbacks", ["matched_parameter"]
    )
    op.create_index("ix_ai_thp_fallbacks_run_item_id", "ai_thp_fallbacks", ["run_item_id"])


def downgrade() -> None:
    op.drop_index("ix_ai_thp_fallbacks_run_item_id", table_name="ai_thp_fallbacks")
    op.drop_index("ix_ai_thp_fallbacks_matched_parameter", table_name="ai_thp_fallbacks")
    op.drop_index("ix_ai_thp_fallbacks_reason", table_name="ai_thp_fallbacks")
    op.drop_table("ai_thp_fallbacks")
