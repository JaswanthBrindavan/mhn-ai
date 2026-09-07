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
``prescriptions``, ``vaccinations`` or ``bills``), writes the row's ``content``, and DELETEs
the source ``unclassified_files`` row. The THP tables below are read-only.

The **only other** Spring column this service writes is ``user.aliases`` (2026-09-07), when
a user claims a name-mismatched document as their own. ``user`` is otherwise read-only, and
that one exception is called out on the table itself. It never issues DDL against any Spring
table.
"""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
)
from sqlalchemy.dialects.postgresql import ARRAY, ENUM, JSONB, UUID

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

#: The account holder. This service resolves whose document it is holding so it can check
#: the name printed on it against the account's. It makes no access decision from this
#: table — family access is Spring's, and stays Spring's (see app/api/deps.py). Bound under
#: its real name, ``user``, which is a reserved word in Postgres; SQLAlchemy quotes it. The
#: Python name is plural so it cannot shadow anything.
#:
#: **``aliases`` is the one column here this service WRITES** (Spring's ``V50``, 2026-09-07)
#: — appended when a user claims a name-mismatched document as their own, so the question is
#: asked once per name instead of once per document. ``id`` and ``name`` remain read-only,
#: and this is only the second Spring-owned table we write at all; filing is the other.
#: Nullable with no default, so every read and write coalesces it to an empty array.
users = Table(
    "user",
    spring_metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("name", String(255), nullable=False),
    Column("aliases", ARRAY(String(255)), nullable=True),
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
    # The user's own filename, carried across at filing. Before this column existed
    # every mover dropped it -- ours and Spring's alike.
    Column("name", String(255), nullable=True),
    # The date printed on the document, chosen by app/services/document_date.py.
    # NOT created_at, which is the moment this service filed the row.
    Column("date", DateTime(timezone=True), nullable=True),
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
    # The user's own filename, carried across at filing. Before this column existed
    # every mover dropped it -- ours and Spring's alike.
    Column("name", String(255), nullable=True),
    # The date printed on the document, chosen by app/services/document_date.py.
    # NOT created_at, which is the moment this service filed the row.
    Column("date", DateTime(timezone=True), nullable=True),
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
    # The user's own filename, carried across at filing. Before this column existed
    # every mover dropped it -- ours and Spring's alike.
    Column("name", String(255), nullable=True),
    # The date printed on the document, chosen by app/services/document_date.py.
    # NOT created_at, which is the moment this service filed the row.
    Column("date", DateTime(timezone=True), nullable=True),
    # The policy's own period. `to_date` is what a renewal alarm is computed from, and
    # both were null on every policy ever filed until filing began writing them
    # (2026-08-31) -- the values had been sitting in `content` all along.
    Column("from_date", DateTime(timezone=True), nullable=True),
    Column("to_date", DateTime(timezone=True), nullable=True),
    # numeric(12, 2), wider than the bills pair: a sum insured of one crore is eight
    # digits before a policy is unusual, and an over-ceiling value is skipped rather than
    # truncated -- so too narrow a column empties exactly the largest policies.
    Column("sum_insured", Numeric(12, 2), nullable=True),
    Column("premium", Numeric(12, 2), nullable=True),
    # One currency for BOTH amounts, as bills does it: a policy prints one.
    Column(
        "amount_currency",
        ENUM("INR", "USD", "EUR", "GBP", name="currency_enum", create_type=False),
        nullable=True,
    ),
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
    # The user's own filename, carried across at filing. Before this column existed
    # every mover dropped it -- ours and Spring's alike.
    Column("name", String(255), nullable=True),
    # The date printed on the document, chosen by app/services/document_date.py.
    # NOT created_at, which is the moment this service filed the row.
    Column("date", DateTime(timezone=True), nullable=True),
)

#: The bills section. Unlike the sections above, this one has columns for the very things
#: we extract — `amount`, `amount_due` and `amount_currency` — so `filing.extra_columns`
#: fills them, the way it fills `vaccinations.next_due_on`. `hospital` stays null for the
#: same reason as `insurance.provider`: it is an FK needing a name-to-id lookup.
#:
#: `amount_currency` is Postgres's `currency_enum`, declared as an ENUM rather than a
#: String because psycopg sends a String parameter as text and Postgres will not implicitly
#: cast text into an enum on INSERT. `create_type=False` keeps this a binding, never DDL —
#: the type belongs to Spring's baseline migration.
bills = Table(
    "bills",
    spring_metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", UUID(as_uuid=True), nullable=False),
    Column("created_by", UUID(as_uuid=True), nullable=True),
    Column("filepath", String(500), nullable=False),
    Column("content", JSONB, nullable=True),
    Column("private", Boolean, nullable=True),
    # The user's own filename, carried across at filing. Before this column existed
    # every mover dropped it -- ours and Spring's alike.
    Column("name", String(255), nullable=True),
    # The date printed on the document, chosen by app/services/document_date.py.
    # NOT created_at, which is the moment this service filed the row.
    Column("date", DateTime(timezone=True), nullable=True),
    Column("amount", Numeric(10, 2), nullable=True),
    Column("amount_due", Numeric(10, 2), nullable=True),
    Column(
        "amount_currency",
        ENUM("INR", "USD", "EUR", "GBP", name="currency_enum", create_type=False),
        nullable=True,
    ),
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
    # The user's own filename, carried across at filing. Before this column existed
    # every mover dropped it -- ours and Spring's alike.
    Column("name", String(255), nullable=True),
    # The date printed on the document, chosen by app/services/document_date.py.
    # NOT created_at, which is the moment this service filed the row.
    Column("date", DateTime(timezone=True), nullable=True),
    Column("next_due_on", DateTime(timezone=True), nullable=True),
    # Which dose this record is, as printed: "Booster", "Dose 2 of 3", "Td". Free text at
    # 128 to match the extraction's own cap on `dose_info` -- an integer column would
    # force a parse that silently drops three of those four.
    Column("dose", String(128), nullable=True),
)

# --- Staff-dashboard THP tables (read-only) ---------------------------------
# The R&D team curates doctor-approved traditional health parameters (THPs), their ideal
# ranges per age bracket, and the alternate units a lab might print, via a staff dashboard.
# The rows land in these Spring-owned tables and we read them to override a report's
# printed reference range (see app/services/ideal_ranges).
#
# Shapes follow Spring's V14 (the staff-dashboard workflow columns) and V18 (the curated
# catalogue: 193 parameters, 1184 aliases, 277 age/sex ranges). Only the columns we read
# are declared: thp_age_range also carries min/ideal/max, which drive the dashboard's gauge
# but not our three-value abnormal flag.
#
# V14 moved three things this service depends on, and the old shapes are still present in
# the schema, so reading the wrong one fails silently rather than erroring:
#   * approval is ``status``, not the ``approved`` boolean -- which V1 created, V18 never
#     sets, and nothing in Spring writes. It is left undeclared deliberately: a binding
#     for it is a binding for a column that is false on every curated row.
#   * aliases live in ``thp_alias``, not the inline ``aliases varchar(100)[]`` array, which
#     V18 leaves null on all 193 rows. Also left undeclared, for the same reason.
#   * an age bracket is per (parameter, SEX) -- see the note on thp_age_range below.

#: Spring's staff-dashboard workflow states (V14). Declared as the real enum rather than a
#: string: PostgreSQL will not compare ``reference_status_enum`` with a bound varchar, so a
#: String column turns every status filter into a runtime error.
_REFERENCE_STATUS = ENUM(
    "draft",
    "pending",
    "approved",
    "rejected",
    "archived",
    "merged",
    name="reference_status_enum",
    create_type=False,
)

#: THP master. ``status`` is the doctor-approval gate (the reference_status_enum:
#: draft/pending/approved/rejected/archived/merged). ``visible`` is NOT declared: the
#: dashboard labels it "Customer visibility?", so it governs app display, not approval.
#: ``ai_integrated`` is the dashboard's own switch for whether a parameter takes part in AI
#: processing at all; nothing in Spring writes it yet and it defaults true, so honouring it
#: costs nothing today and means a staff member turning it off is not silently ignored.
traditional_health_parameters = Table(
    "traditional_health_parameters",
    spring_metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(100), nullable=False),
    #: The unit the parameter's ideal ranges are curated in.
    Column("units", String(25), nullable=False),
    Column("status", _REFERENCE_STATUS, nullable=False),
    Column("ai_integrated", Boolean, nullable=False),
    #: Soft delete. A deleted parameter must not go on matching test names.
    Column("deleted_at", DateTime(timezone=True), nullable=True),
)

#: Other names the same parameter appears under on a report. Curated per row rather than as
#: an array since V14, with its own approval state -- an alias may be a staff entry or an
#: unreviewed OCR/AI suggestion, and only an approved one may decide a patient's flag.
thp_alias = Table(
    "thp_alias",
    spring_metadata,
    Column("thp_id", Integer, nullable=True),
    Column("alias", String(150), nullable=False),
    Column("status", _REFERENCE_STATUS, nullable=False),
)

#: Ideal range per age bracket, inclusive on both ends. ``low_warn``/``high_warn`` bound
#: the dashboard's "Ideal Range"; see _IDEAL_FLOOR in app/services/ideal_ranges.py.
#:
#: ``sex`` is 'any' | 'male' | 'female' and is part of the row's identity (V14 re-made the
#: unique index as (thp_id, sex, age_min, age_max)). 78 of the 277 curated ranges are
#: sex-specific, so a bracket picked without reading this column can be the other sex's.
thp_age_range = Table(
    "thp_age_range",
    spring_metadata,
    Column("thp_id", Integer, nullable=False),
    Column("sex", String(8), nullable=False),
    Column("age_min", Integer, nullable=False),
    Column("age_max", Integer, nullable=False),
    Column("low_warn", Float, nullable=False),
    Column("high_warn", Float, nullable=False),
    Column("status", _REFERENCE_STATUS, nullable=False),
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
