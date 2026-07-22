"""Read-only access to Spring-owned tables.

Deliberately defined on a **separate MetaData**, not on ``app.core.db.Base``. Anything
registered on ``Base`` becomes a migration target; these tables belong to the Spring
backend and this service must never create, alter, or drop them. Keeping them off
``Base.metadata`` makes that structurally impossible rather than merely discouraged.

Only the columns this service actually reads are declared. That is intentional: a
partial declaration cannot drift into being mistaken for the authoritative schema.

**Read-only.** No INSERT, UPDATE, or DELETE against these tables, with one documented
exception added later — writing the assembled AI payload into ``reports.content`` via a
JSONB merge, which is the agreed hand-off point with Spring.
"""

from sqlalchemy import Boolean, Column, Integer, MetaData, String, Table
from sqlalchemy.dialects.postgresql import JSONB, UUID

spring_metadata = MetaData()

reports = Table(
    "reports",
    spring_metadata,
    # NOTE: integer, not UUID. Verified against the live database.
    Column("id", Integer, primary_key=True),
    # Whose report it is (the subject) -- NOT necessarily who uploaded it.
    Column("user_id", UUID(as_uuid=True), nullable=False),
    # Who uploaded it. Differs from user_id for family uploads.
    Column("created_by", UUID(as_uuid=True), nullable=True),
    # The S3 object key.
    Column("filepath", String(500), nullable=False),
    Column("content", JSONB, nullable=True),
    Column("private", Boolean, nullable=True),
)
