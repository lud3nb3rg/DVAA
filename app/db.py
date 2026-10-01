"""Database layer.

Structured app state uses SQLModel. A handful of challenges deliberately reach
past this into raw SQL (see `raw_connection`) because unparameterized queries
are the vulnerability being taught.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import Index
from sqlmodel import Field, Session, SQLModel, create_engine

from app.config import settings


def _utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------
# Framework tables
# --------------------------------------------------------------------------


class ChallengeState(SQLModel, table=True):
    """Per-session, per-challenge difficulty level and scratch state.

    One row per (session, challenge), enforced in the schema. Every writer here
    reads the row and then writes it, and two clicks arriving together would
    otherwise both find nothing and both insert - after which the reader's
    `.first()` picks whichever row the database hands back and half the writes
    vanish.
    """

    __tablename__ = "challenge_state"
    __table_args__ = (
        Index("uq_challenge_state_session_challenge", "session_id", "challenge_id", unique=True),
    )

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    challenge_id: str = Field(index=True)
    level: str = "low"
    state_json: str = "{}"
    updated_at: datetime = Field(default_factory=_utcnow)


class Solve(SQLModel, table=True):
    """A captured flag.

    Unique per (session, challenge, level): `record_solve` checks before it
    inserts, and without the constraint two submissions of the same flag racing
    each other both pass the check and both score.
    """

    __tablename__ = "solves"
    __table_args__ = (
        Index("uq_solves_session_challenge_level", "session_id", "challenge_id", "level", unique=True),
    )

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    player: str = "anonymous"
    challenge_id: str = Field(index=True)
    level: str
    solved_at: datetime = Field(default_factory=_utcnow)


class Attempt(SQLModel, table=True):
    """How many times a player has run a challenge at a level, and since when.

    A leaderboard is dead weight when you are the only person on the lab, so
    the run summary scores you against your own run instead: how many tries a
    capture took, and how long it sat unsolved. That needs a count and a first
    timestamp, and nothing else.

    One row per (session, challenge, level), enforced in the schema for the same
    reason as the others: the recorder reads then writes.
    """

    __tablename__ = "attempts"
    __table_args__ = (
        Index(
            "uq_attempts_session_challenge_level",
            "session_id",
            "challenge_id",
            "level",
            unique=True,
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    challenge_id: str = Field(index=True)
    level: str = "low"
    count: int = 0
    first_at: datetime = Field(default_factory=_utcnow)
    last_at: datetime = Field(default_factory=_utcnow)


class Setting(SQLModel, table=True):
    """Small key/value store for the lab's own bookkeeping.

    Currently holds the seed version, so a database written by an older build
    can tell that its fixture rows are stale.
    """

    __tablename__ = "settings"

    key: str = Field(primary_key=True)
    value: str = ""


class AgentMemory(SQLModel, table=True):
    """Persistent agent memory.

    NOTE: intentionally NOT scoped by session. That is the LLM05 vulnerability:
    one poisoned entry reaches every future session that reads the store.
    """

    __tablename__ = "agent_memory"

    id: int | None = Field(default=None, primary_key=True)
    namespace: str = Field(index=True)
    content: str
    source: str = "user"
    approved: bool = True
    created_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------
# Fixture tables (seeded scenario data)
# --------------------------------------------------------------------------


class Ticket(SQLModel, table=True):
    """Customer support tickets - the LLM01 indirect injection surface.

    `reply` is what the triage assistant sends back down the thread to whoever
    filed the ticket, which is where an injection's output actually lands.
    """

    __tablename__ = "tickets"

    id: int | None = Field(default=None, primary_key=True)
    subject: str
    body: str
    submitter: str
    staff_note: str = ""
    reply: str = ""
    status: str = "open"
    # True for tickets the player filed, false for the seeded queue. Only your
    # own thread shows you its replies.
    from_player: bool = False
    # Which player session filed it. `from_player` alone says a ticket belongs
    # to *a* player, not to *you* - and the challenge tells the learner they can
    # only read threads they opened, so that has to hold against another player
    # on the same lab, not only against the seeded queue. Empty when seeded.
    owner_session: str = Field(default="", index=True)
    created_at: datetime = Field(default_factory=_utcnow)


class Email(SQLModel, table=True):
    """Mailbox for the LLM03 email assistant."""

    __tablename__ = "emails"

    id: int | None = Field(default=None, primary_key=True)
    mailbox: str = Field(index=True)
    sender: str
    recipient: str
    subject: str
    body: str
    folder: str = "inbox"
    # True for mail the player sent into the mailbox themselves - their own
    # indirect-injection payload - so a scenario reset can withdraw it without
    # touching the seeded correspondence.
    from_player: bool = False
    created_at: datetime = Field(default_factory=_utcnow)


class HrRecord(SQLModel, table=True):
    """Employee records for the LLM02 HR assistant. All data is fictional."""

    __tablename__ = "hr_records"

    id: int | None = Field(default=None, primary_key=True)
    employee: str = Field(index=True)
    role: str
    manager: str
    salary_band: str
    national_id: str  # fictional format
    case_ref: str = ""  # sealed HR case reference; the LLM02 canary
    performance_note: str


class Document(SQLModel, table=True):
    """Knowledge-base documents for the LLM09 multi-tenant RAG challenge."""

    __tablename__ = "documents"

    id: int | None = Field(default=None, primary_key=True)
    tenant: str = Field(index=True)
    title: str
    body: str
    classification: str = "internal"
    source: str = "seed"
    created_at: datetime = Field(default_factory=_utcnow)


class RegistryArtifact(SQLModel, table=True):
    """Mock package-registry entries for the LLM04 supply-chain surrogate."""

    __tablename__ = "registry_artifacts"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    publisher: str
    version: str
    fmt: str  # "pickle" | "safetensors"
    signature: str = ""
    signed_by: str = ""
    description: str = ""
    # The artifact's own dependencies, as the requirement strings it declares:
    # "tensor-utils==0.9.4". The install job resolves these after loading it,
    # which is where the transitive gap at High lives.
    requires: str = ""
    # What the artifact's payload does when it is deserialized, and whether the
    # player published it themselves.
    payload_kind: str = ""
    from_player: bool = False
    # Retained so lab databases written by an earlier build still satisfy their
    # NOT NULL column. Nothing reads it: the registry publishes its own computed
    # digest, and Impossible pins that digest in a lockfile.
    declared_sha256: str = ""
    payload_path: str = ""


# --------------------------------------------------------------------------
# Engine / session helpers
# --------------------------------------------------------------------------

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        settings.db_file.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{settings.db_file}",
            connect_args={"check_same_thread": False},
        )
    return _engine


def init_db() -> None:
    engine = get_engine()
    SQLModel.metadata.create_all(engine)
    _add_missing_columns(engine)
    _add_missing_indexes(engine)


# Tables whose uniqueness is enforced by an index, and the columns it covers.
# `create_all` builds these for a fresh database; `_add_missing_indexes` is how
# a lab database written before they existed catches up.
_UNIQUE_INDEXES = {
    "uq_challenge_state_session_challenge": ("challenge_state", ("session_id", "challenge_id")),
    "uq_solves_session_challenge_level": ("solves", ("session_id", "challenge_id", "level")),
    "uq_attempts_session_challenge_level": (
        "attempts",
        ("session_id", "challenge_id", "level"),
    ),
}


def _add_missing_indexes(engine) -> None:
    """Add the unique indexes, dropping any duplicates already recorded.

    An existing lab database can hold duplicate rows written before the
    constraint existed - `CREATE UNIQUE INDEX` would fail on them and take the
    whole boot with it. The extras are redundant by definition (same session,
    same challenge, same level), so the lowest id wins and the rest go.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for index_name, (table, columns) in _UNIQUE_INDEXES.items():
            if table not in existing_tables:
                continue
            if index_name in {i["name"] for i in inspector.get_indexes(table)}:
                continue
            key = ", ".join(columns)
            conn.execute(
                text(
                    f"DELETE FROM {table} WHERE id NOT IN "
                    f"(SELECT MIN(id) FROM {table} GROUP BY {key})"
                )
            )
            conn.execute(
                text(f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table} ({key})")
            )


def _add_missing_columns(engine) -> None:
    """Bring an existing lab database up to the current models.

    DVAA's database is fixture data, not user data, but a workshop machine that
    already ran an older build should not need it deleted by hand just because a
    scenario gained a column. `create_all` never alters an existing table, so
    do the one thing SQLite supports cheaply: append the columns that are new.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                ddl = f"ALTER TABLE {table.name} ADD COLUMN {column.name} "
                ddl += column.type.compile(engine.dialect)
                default = getattr(column.default, "arg", None)
                if isinstance(default, bool):
                    ddl += f" NOT NULL DEFAULT {int(default)}"
                elif isinstance(default, int | float):
                    ddl += f" NOT NULL DEFAULT {default}"
                elif isinstance(default, str):
                    ddl += " NOT NULL DEFAULT '{}'".format(default.replace("'", "''"))
                conn.execute(text(ddl))


@contextmanager
def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session


@contextmanager
def raw_connection() -> Iterator[sqlite3.Connection]:
    """A raw DB-API connection, opened READ-ONLY.

    Used by challenges that must demonstrate unparameterized SQL against the
    whole database - LLM03's leftover `run_report` tool runs whatever query the
    model hands it. Do not imitate this pattern in real code.

    Read-only because the lesson is what an over-permissioned query can *read*,
    not what it can destroy. A read-write handle here is not a stronger version
    of that lesson, it is a way to lose the lab: `sqlite3` rolls an uncommitted
    DELETE back when the connection closes, but DDL runs in autocommit, so one
    injected `DROP TABLE solves` would take the player's progress and the seeded
    world with it permanently - collateral that teaches nothing the SELECT does
    not already teach. `mode=ro` leaves every table readable and every join
    reachable, and turns a write into an error the tool reports like any other.
    """
    # `as_uri()` rather than string-building: DVAA_DB_PATH is operator-supplied,
    # and sqlite reads a bare '#' in a URI as the start of a fragment, which
    # would silently open (and create) a different database.
    conn = sqlite3.connect(f"{settings.db_file.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def reset_progress(session_id: str) -> None:
    """Clear a player's solves and per-challenge state."""
    from sqlmodel import delete

    with get_session() as s:
        s.exec(delete(Solve).where(Solve.session_id == session_id))
        s.exec(delete(ChallengeState).where(ChallengeState.session_id == session_id))
        s.exec(delete(Attempt).where(Attempt.session_id == session_id))
        s.commit()
