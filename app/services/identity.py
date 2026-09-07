"""Whether a document's printed name belongs to the account it is being filed into.

The one module that joins the name matcher to the database. ``app.services.names`` stays
pure — its table of cases is its whole contract — so everything needing a session lives
here instead.

**It reads ``user.name`` and ``unclassified_files``, and nothing else about people.** It
does not read ``family_connect`` and makes no decision about who may access whose records.
Family membership and write access are the Spring backend's decisions and stay there; when
a candidate list is needed, Spring hands us one already filtered (see
``names.matches_any``). The rule in ``app/api/deps.py`` holds: a second implementation of
the family rules here would drift, and a drift bug leaks one family member's records to
another.

**It also WRITES one column, ``user.aliases``** (2026-09-07), and that is the only Spring
column this service writes outside filing. Confirming a mismatched document records the
name on it, so the same name is never questioned again: ``identity_confirmed_at`` is keyed
on the document, and in production one name was confirmed 82 times across 87 documents.

An alias widens what counts as this account holder, permanently and silently, so two
properties matter. It is only ever written from ``confirm_identity`` — the user claiming a
document as **theirs** — never from a reassignment, which says the opposite. And it is
still not an access decision: it says which names this account answers to, never who may
read whose records.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import NamedTuple

from sqlalchemy import String, func, select, update
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.orm import Session

from app.models.ai_results import AiReportClassification
from app.models.spring import unclassified_files, users
from app.services import filing
from app.services.classification import DocumentSection
from app.services.names import NameVerdict, compare_all, normalise
from app.workers.stagetypes import RejectStageError, StageContext

logger = logging.getLogger(__name__)

#: Typed, so Postgres can resolve ``array_append``'s element type when the column is NULL.
#: A bare empty-array literal there is ambiguous and the statement fails to plan.
_EMPTY_ALIASES = array([], type_=String(255))


class Owner(NamedTuple):
    """Every name an account answers to: the one on the account, plus confirmed aliases."""

    user_id: uuid.UUID
    name: str
    aliases: list[str]

    @property
    def all_names(self) -> list[str]:
        return [self.name, *self.aliases]


def owner(session: Session, document_id: int) -> Owner | None:
    """The account this document is being filed into, and the names it answers to.

    None when the intake row is gone — which is normal, not an error: filing deletes it,
    so every retry of an already-filed document lands here. The caller treats that as
    "already settled" rather than as a failure.

    ``aliases`` are names the user has previously confirmed on their own documents. They
    are matched exactly as the account's own name is, which is the point: the question is
    asked once per name rather than once per document. A null column reads as an empty
    list — Spring's ``V50`` adds it nullable with no default.
    """
    row = session.execute(
        select(users.c.id, users.c.name, users.c.aliases)
        .select_from(unclassified_files.join(users, unclassified_files.c.user_id == users.c.id))
        .where(unclassified_files.c.id == document_id)
    ).one_or_none()
    if row is None:
        return None
    return Owner(row.id, str(row.name), [str(a) for a in (row.aliases or [])])


def owner_name(session: Session, document_id: int) -> str | None:
    """The name on the account this document is being filed into, ignoring aliases.

    Kept for callers that want only the account's own name. The gate uses ``owner`` — it
    must compare against the aliases too.
    """
    found = owner(session, document_id)
    return found.name if found is not None else None


class SettledName(NamedTuple):
    """What was actually stored about a document's name, verdict and confirmation alike."""

    name_match: str | None
    patient_name: str | None
    identity_confirmed_at: datetime | None


def settled_row(session: Session, document_id: int) -> SettledName | None:
    """The classification row a settled verdict comes from, unmassaged.

    **A confirmation is checked across ALL rows, not just the newest one.** A retry inserts
    a fresh classification with a null verdict, so reading only the latest row would report
    "nothing settled" for a document the user has already claimed as theirs — and the gate
    would ask them again. Re-prompting someone about a decision they already made is the
    reflex-dismissal failure this whole feature exists to avoid. The computed verdict below
    it stays latest-wins: that one is a reading of the document and the newest reading is
    the current one.

    Separate from ``settled_verdict`` because that function answers a routing question and
    coerces a confirmed mismatch to MATCH to answer it. The stored truth — "mismatch, and
    the user said it is theirs" — is what the payload has to carry, and it is not
    recoverable from the coerced answer.
    """
    rows = session.execute(
        select(
            AiReportClassification.name_match,
            AiReportClassification.patient_name,
            AiReportClassification.identity_confirmed_at,
        )
        .where(AiReportClassification.document_id == document_id)
        .order_by(AiReportClassification.created_at.desc())
    ).all()
    if not rows:
        return None
    confirmed = next((r for r in rows if r.identity_confirmed_at is not None), rows[0])
    return SettledName(*confirmed)


def _verdict_of(row: SettledName) -> NameVerdict | None:
    if row.identity_confirmed_at is not None:
        return NameVerdict.MATCH
    return NameVerdict(row.name_match) if row.name_match is not None else None


def settled_verdict(session: Session, document_id: int) -> NameVerdict | None:
    """The verdict already reached for this document, across every run item.

    Keyed on document_id, not run item: a retry mints a new item, and a decision the user
    made must outlive it. A confirmed identity outranks whatever was computed.
    """
    row = settled_row(session, document_id)
    return _verdict_of(row) if row is not None else None


def record_verdict(session: Session, item_id: uuid.UUID, verdict: NameVerdict) -> None:
    """Store the verdict computed for one run item's classification."""
    session.execute(
        update(AiReportClassification)
        .where(AiReportClassification.run_item_id == item_id)
        .values(name_match=verdict.value)
    )
    session.commit()


def _carry_settled(session: Session, item_id: uuid.UUID, settled: SettledName) -> None:
    """Copy an already-settled name state onto THIS item's classification row.

    Confirming an identity re-submits the document as a NEW run item with a new, blank
    classification. The gate then short-circuits on the settled verdict without ever
    reaching ``record_verdict`` — so the row the payload is built from keeps a null
    verdict, and both ``content.ai.name_check`` and ``/status.name_check`` come back null
    for every document the user has claimed as their own. The standing warning the app
    shows on such a document ("the name doesn't match your profile") then never renders.

    What is written is the STORED state, not the gate's routing answer: ``mismatch`` with
    its confirmation stamp. Writing ``match`` here would satisfy the same null check and
    leave the warning just as unrenderable.
    """
    session.execute(
        update(AiReportClassification)
        .where(AiReportClassification.run_item_id == item_id)
        .values(
            name_match=settled.name_match,
            patient_name=settled.patient_name,
            identity_confirmed_at=settled.identity_confirmed_at,
        )
    )
    session.commit()


def learn_alias(session: Session, *, document_id: int, printed: str | None) -> bool:
    """Remember that this account answers to ``printed``, so it is never asked again.

    The write half of the aliases feature, and **the only column of ``user`` this service
    writes**. Everything else it reads there is read-only, so keep this the one place.

    Does nothing when: there is no readable name (nothing to remember); the intake row is
    gone (a name mismatch is never filed, so the row is always there for the case this
    serves — its absence means something else, and inventing an alias from it would be a
    guess); or the name already matches one the account holds, which covers both a repeat
    confirmation and a variant close enough that ``compare`` already accepts it. Matching
    rather than string equality is what stops "P SURESH BABU" and "P Suresh Babu"
    accumulating as two entries.

    Returns whether an alias was actually added, for the caller's log.
    """
    if not normalise(printed):
        return False
    account = owner(session, document_id)
    if account is None:
        return False
    if compare_all(printed, account.all_names) is NameVerdict.MATCH:
        return False

    name = str(printed).strip()[:255]
    session.execute(
        update(users)
        .where(users.c.id == account.user_id)
        .values(aliases=func.array_append(func.coalesce(users.c.aliases, _EMPTY_ALIASES), name))
    )
    logger.info(
        "name_alias_learned",
        extra={"document_id": document_id, "alias_count": len(account.aliases) + 1},
    )
    return True


def confirm_identity(session: Session, document_id: int) -> bool:
    """Record that the user claimed a mismatched document as their own.

    Stamps the most recent classification for the document, **and remembers the name** so
    the next document printing it is not questioned again. False when there is nothing to
    stamp, which the caller turns into a 409.

    The stamp and the alias are one transaction on purpose. Stamping without learning
    leaves the user answering the same question for ever, which is the bug this fixes;
    learning without stamping would accept the name while still refusing this document.
    """
    latest = session.execute(
        select(AiReportClassification.id, AiReportClassification.patient_name)
        .where(AiReportClassification.document_id == document_id)
        .order_by(AiReportClassification.created_at.desc())
        .limit(1)
    ).one_or_none()
    if latest is None:
        return False
    session.execute(
        update(AiReportClassification)
        .where(AiReportClassification.id == latest.id)
        .values(identity_confirmed_at=datetime.now(UTC))
    )
    learn_alias(session, document_id=document_id, printed=latest.patient_name)
    session.commit()
    return True


def gate(ctx: StageContext, section: DocumentSection) -> None:
    """Refuse to file a document into a wallet whose owner's name it does not carry.

    Called after the section is known and BEFORE filing, so a refusal costs nothing to
    undo: the document keeps its intake row and its object, and the app's delete is one
    row and one key. Filing is never reversed in this service, which is exactly why this
    check cannot happen after it.

    Passes silently — and deliberately — in four cases:

    * the feature is off;
    * the section is one this service never files, so the document is being rejected for
      that reason anyway and asking the user about identity would be noise;
    * a verdict was already settled (matched before, or the user confirmed it). A retry
      must not re-interrogate someone — but it does carry that settled state onto its own
      classification row, or the document's payload would describe no verdict at all;
    * the intake row is gone. That is a retry of an already-filed document, and filing
      owns the missing-source case — raising here would relabel it as an identity problem.
    """
    if not ctx.settings.name_matching_enabled:
        return
    if section not in filing.SECTION_TABLES:
        return

    settled = settled_row(ctx.session, ctx.document_id)
    if settled is not None and _verdict_of(settled) in (NameVerdict.MATCH, NameVerdict.UNKNOWN):
        _carry_settled(ctx.session, ctx.item_id, settled)
        return

    account = owner(ctx.session, ctx.document_id)
    if account is None:
        return

    printed = ctx.session.execute(
        select(AiReportClassification.patient_name).where(
            AiReportClassification.run_item_id == ctx.item_id
        )
    ).scalar_one_or_none()

    # Against the account's own name AND every alias it has confirmed. Without the
    # aliases this asked again on every document: in production one name was confirmed
    # 82 times across 87 documents, and a dialog answered that often stops being read.
    verdict = compare_all(printed, account.all_names)
    record_verdict(ctx.session, ctx.item_id, verdict)
    if verdict is NameVerdict.MISMATCH:
        logger.info(
            "document_name_mismatch",
            extra={"item_id": str(ctx.item_id), "document_id": ctx.document_id},
        )
        raise RejectStageError(
            "name_mismatch",
            "The name on this document does not match the account it was uploaded to",
        )
