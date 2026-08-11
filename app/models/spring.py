"""Access to Spring-owned tables.

Deliberately defined on a **separate MetaData**, not on ``app.core.db.Base``. Anything
registered on ``Base`` becomes a migration target; these tables belong to the Spring
backend and this service must never create, alter, or drop them (no DDL). Keeping them
off ``Base.metadata`` makes that structurally impossible rather than merely discouraged.

Only the columns this service uses are declared. That is intentional: a partial
declaration cannot drift into being mistaken for the authoritative schema.

**Row access, not schema.** ``unclassified_files`` is read-only except for the documented
filing move: once a document is classified into a section this service processes, it
INSERTs a row into **that section's table** (``reports``, ``scans_imaging``, ``insurance``,
``prescriptions`` or ``vaccinations``), writes the row's ``content``, and DELETEs the
source ``unclassified_files`` row. The THP tables below are read-only. It never issues DDL
against any Spring table.
"""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

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

#: The scans/imaging section. INSERTed into when a scan is filed; read otherwise.
scans_imaging = Table(
    "scans_imaging",
    spring_metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", UUID(as_uuid=True), nullable=False),
    Column("created_by", UUID(as_uuid=True), nullable=True),
    Column("filepath", String(500), nullable=False),
    Column("content", JSONB, nullable=True),
    Column("private", Boolean, nullable=True),
)

#: The insurance section. NOTE: its FK column is `provider` (-> insurance_provider), not
#: `hospital`. We leave it null — resolving a provider name to an id is separate work.
insurance = Table(
    "insurance",
    spring_metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", UUID(as_uuid=True), nullable=False),
    Column("created_by", UUID(as_uuid=True), nullable=True),
    Column("filepath", String(500), nullable=True),
    Column("content", JSONB, nullable=True),
    Column("private", Boolean, nullable=True),
)

#: The prescriptions section. INSERTed into when a prescription is filed; read otherwise.
#:
#: NOTE: its FK column is `hospital` (-> a hospital master), and we leave it null for the
#: same reason as `insurance.provider` — resolving a hospital name to an id is a lookup
#: this service does not do. The prescriber's name is not lost by that: it is transcribed
#: into `content.ai.section_extraction.fields.prescriber` as printed.
prescriptions = Table(
    "prescriptions",
    spring_metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", UUID(as_uuid=True), nullable=False),
    Column("created_by", UUID(as_uuid=True), nullable=True),
    Column("filepath", String(500), nullable=False),
    Column("content", JSONB, nullable=True),
    Column("private", Boolean, nullable=True),
)

#: The vaccinations section. `next_due_on` drives Spring's "vaccination due" index and is
#: the one extra column we can fill, from the extracted next_due_date.
vaccinations = Table(
    "vaccinations",
    spring_metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", UUID(as_uuid=True), nullable=False),
    Column("created_by", UUID(as_uuid=True), nullable=True),
    Column("filepath", String(500), nullable=False),
    Column("content", JSONB, nullable=True),
    Column("private", Boolean, nullable=True),
    Column("next_due_on", DateTime(timezone=True), nullable=True),
)

# --- Staff-dashboard THP tables (read-only) ---------------------------------
# The R&D team curates doctor-approved traditional health parameters (THPs), their ideal
# ranges per age bracket, and the alternate units a lab might print, via a staff dashboard.
# The rows land in these Spring-owned tables and we read them to override a report's
# printed reference range (see app/services/ideal_ranges).
#
# Shapes follow the Spring migration of 2026-07-30. Only the columns we read are declared:
# thp_age_range also carries min/low_danger/ideal/high_danger/max, which drive the
# dashboard's gauge but not our three-value abnormal flag.

#: THP master. ``approved`` is the doctor-approval gate. ``visible`` is NOT declared: the
#: dashboard labels it "Customer visibility?", so it governs app display, not approval.
traditional_health_parameters = Table(
    "traditional_health_parameters",
    spring_metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(100), nullable=False),
    #: The unit the parameter's ideal ranges are curated in.
    Column("units", String(25), nullable=False),
    Column("approved", Boolean, nullable=True),
    #: Other names the same parameter appears under on a report.
    Column("aliases", ARRAY(String(100)), nullable=True),
)

#: Ideal range per age bracket, inclusive on both ends. ``low_warn``/``high_warn`` bound
#: the dashboard's "Ideal Range"; see _IDEAL_FLOOR in app/services/ideal_ranges.py.
thp_age_range = Table(
    "thp_age_range",
    spring_metadata,
    Column("thp_id", Integer, nullable=False),
    Column("age_min", Integer, nullable=False),
    Column("age_max", Integer, nullable=False),
    Column("low_warn", Float, nullable=False),
    Column("high_warn", Float, nullable=False),
)

#: Units other than the parameter's own that a report may print, with the conversion into
#: it: ``base = printed * multiplier + offset_value``.
thp_alternate_units = Table(
    "thp_alternate_units",
    spring_metadata,
    Column("thp_id", Integer, nullable=False),
    Column("name", String(100), nullable=False),
    Column("multiplier", Float, nullable=False),
    Column("offset_value", Float, nullable=False),
)
