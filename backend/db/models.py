"""
Neon tables.

A chat owns exactly one document. Its vectors live in the owning user's
Pinecone namespace under ids prefixed with the chat id.

`drs_chat_sessions` is the chat itself, not a session layer on top of one —
the name predates the current model. It is the only home for bm25_params,
ownership and ingest status, so it cannot be derived from the messages table.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import (Column, DateTime, Index, Integer, LargeBinary,
                        String, Text)
from sqlalchemy.dialects.postgresql import JSONB

from db.database import Base

IST = timezone(timedelta(hours=5, minutes=30))


def now_ist():
    return datetime.now(IST)


class Account(Base):
    __tablename__ = "drs_accounts"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=now_ist)


class LoginFailure(Base):
    """One row per failed login. DB-backed rather than in-memory so the lockout
    holds across API instances and survives the free tier spinning down."""

    __tablename__ = "drs_login_failures"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=now_ist)


class ChatSession(Base):
    """One chat, owning one uploaded document.

    status drives the whole upload UX, because ingestion takes ~45s:
        awaiting_document -> processing -> ready
                                        -> failed
    """

    __tablename__ = "drs_chat_sessions"

    id = Column(String, primary_key=True)            # uuid4 hex, also the id prefix
    user_id = Column(Integer, index=True, nullable=False)
    title = Column(String, default="New Chat")
    status = Column(String, default="awaiting_document", index=True)
    error = Column(Text)                             # failure reason when status='failed'
    stage = Column(String)                           # ingest sub-step while processing (extract/split/chunk/embed/store)

    filename = Column(String)

    # Extraction summary: pages, doc types, chunk count, search label. Lets the
    # sidebar and /structure render without rehydrating the retriever at all.
    doc_stats = Column(JSONB)

    # The fitted BM25Encoder (get_params()), so a session survives a restart
    # with its sparse half intact.
    bm25_params = Column(JSONB)

    created_at = Column(DateTime(timezone=True), default=now_ist)
    updated_at = Column(DateTime(timezone=True), default=now_ist)


class RefreshToken(Base):
    """One row per issued refresh token, stored as a SHA-256 hash so a DB leak
    cannot be replayed. Deleting a row revokes it (that is what /logout does),
    and the row's own expiry is what caps how long a session can be renewed."""

    __tablename__ = "drs_refresh_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True, nullable=False)
    token_hash = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=now_ist)


class ChatMessage(Base):
    """One message. Separate rows rather than a JSON blob so concurrent writes
    cannot clobber each other. Ordered by autoincrement id."""

    __tablename__ = "drs_chat_messages"

    id = Column(Integer, primary_key=True)
    chat_id = Column(String, index=True, nullable=False)
    user_id = Column(Integer, index=True, nullable=False)
    role = Column(String, nullable=False)            # "user" | "assistant"
    content = Column(Text, nullable=False)

    # Citations for an assistant turn, so a reloaded conversation keeps them.
    sources = Column(JSONB)

    created_at = Column(DateTime(timezone=True), default=now_ist)


class IngestJob(Base):
    """One queued ingestion. The queue is this table, claimed by conditional
    UPDATE — no broker, because Neon is already here and a second service is
    not (upgrade_roadmap.txt PART 4 has the Redis line).

    The PDF itself lives in `payload` rather than on disk. A temp-file path
    only works while the worker shares a filesystem with the API, and a job
    whose bytes vanish on restart is not a durable queue. At MAX_UPLOAD_MB the
    row stays small enough for Postgres to hold comfortably.

    status: queued -> running -> done | failed
    A crashed worker leaves a row in 'running' forever, so `claimed_at` is a
    lease: reclaim_stale() returns anything past it to 'queued' while attempts
    remain, and fails it after that. Without the lease a dead worker's job is
    invisible and permanent.
    """

    __tablename__ = "drs_ingest_jobs"

    id = Column(String, primary_key=True)            # uuid4 hex

    # Deduplicates a resubmitted upload. UNIQUE, so a replay loses the
    # insert rather than being waved through by a SELECT that raced it --
    # a double-tap or a client retry after a timeout arrives as two
    # near-simultaneous requests, which is exactly when a check-then-act
    # dedupe fails. Nullable: rows predating this column have none.
    idempotency_key = Column(String, unique=True, index=True)

    chat_id = Column(String, index=True, nullable=False)
    user_id = Column(Integer, index=True, nullable=False)

    filename = Column(String, nullable=False)
    payload = Column(LargeBinary, nullable=False)

    status = Column(String, default="queued", index=True, nullable=False)
    attempts = Column(Integer, default=0, nullable=False)
    max_attempts = Column(Integer, default=3, nullable=False)
    error = Column(Text)

    # Who holds the lease and since when. claimed_by is for debugging only —
    # the claim itself is decided by the UPDATE, never by reading this.
    claimed_by = Column(String)
    claimed_at = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), default=now_ist)
    updated_at = Column(DateTime(timezone=True), default=now_ist)


# The common read is "all messages of one chat, in order".
Index("ix_drs_messages_chat_id_id", ChatMessage.chat_id, ChatMessage.id)
# The common list is "my chats, most recent first".
Index("ix_drs_sessions_user_updated", ChatSession.user_id, ChatSession.updated_at)
# The claim query is "oldest queued job", and the reclaim sweep is
# "running jobs past their lease" — both ride this index.
Index("ix_drs_jobs_status_created", IngestJob.status, IngestJob.created_at)
