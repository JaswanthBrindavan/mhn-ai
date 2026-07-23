"""Access to Spring-owned tables.

Deliberately defined on a **separate MetaData**, not on ``app.core.db.Base``. Anything
registered on ``Base`` becomes a migration target; these tables belong to the Spring
backend and this service must never create, alter, or drop them (no DDL). Keeping them
off ``Base.metadata`` makes that structurally impossible rather than merely discouraged.

Only the columns this service uses are declared. That is intentional: a partial
declaration cannot drift into being mistaken for the authoritative schema.

**Row access, not schema.** These are read-only for the source document
(``unclassified_files``) except for the documented move: when a document is classified as
a report, the service INSERTs a ``reports`` row, writes ``reports.content``, and DELETEs
the source ``unclassified_files`` row. It never issues DDL against any Spring table.
"""

from sqlalchemy import Boolean, Column, DateTime, Integer, MetaData, String, Table
from sqlalchemy.dialects.postgresql import JSONB, UUID

spring_metadata = MetaData()

#: The intake table: every uploaded document lands here first. We read a document's
#: fields to classify it, and delete the row when the document is moved into a section.
unclassified_files = Table(
    "unclassified_files",
    spring_metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", UUID(as_uuid=True), nullable=False),
    Column("filepath", String(500), nullable=False),  # the S3 object key
    Column("private", Boolean, nullable=True),
    Column("created_by", UUID(as_uuid=True), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=True),
    Column("name", String(255), nullable=True),
)

#: The reports section. We INSERT a row here when moving a classified report, and write
#: its ``content``. Read otherwise.
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
