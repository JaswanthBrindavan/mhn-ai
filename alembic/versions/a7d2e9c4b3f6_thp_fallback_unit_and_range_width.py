"""ai_thp_fallbacks: record the printed unit, widen the printed range

The worklist gained a ``unit_mismatch`` reason (the report printed a unit that is neither
the parameter's own nor one of its curated alternates), and the fix R&D has to make is to
add that unit — so the row has to carry it.

``report_reference_range`` was sized 128 before ``ExtractedLabResult.reference_range`` was
raised to 512 (schema ext-3): printed eGFR/HbA1c interpretation scales run 138-160
characters, which would have failed the insert once the feature was switched on.

Revision ID: a7d2e9c4b3f6
Revises: f4b8c2e6a1d9
Create Date: 2026-07-30 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a7d2e9c4b3f6"
# Chained after ai_section_extractions rather than beside it: both were written against
# e3c7a1f5b2d8, and two heads make `alembic upgrade head` fail outright.
down_revision: str | None = "f4b8c2e6a1d9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "ai_thp_fallbacks", sa.Column("report_unit", sa.String(length=64), nullable=True)
    )
    op.alter_column(
        "ai_thp_fallbacks",
        "report_reference_range",
        existing_type=sa.String(length=128),
        type_=sa.String(length=512),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "ai_thp_fallbacks",
        "report_reference_range",
        existing_type=sa.String(length=512),
        type_=sa.String(length=128),
        existing_nullable=True,
    )
    op.drop_column("ai_thp_fallbacks", "report_unit")
