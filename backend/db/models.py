"""Neon tables."""

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
    """One chat, owning one uploaded document."""

    __tablename__ = "drs_chat_sessions"

    id = Column(String, primary_key=True)
    user_id = Column(Integer, index=True, nullable=False)
    title = Column(String, default="New Chat")
    status = Column(String, default="awaiting_document", index=True)
    error = Column(Text)
    stage = Column(String)

    filename = Column(String)

    doc_stats = Column(JSONB)

    bm25_params = Column(JSONB)

    review = Column(JSONB)

    created_at = Column(DateTime(timezone=True), default=now_ist)
    updated_at = Column(DateTime(timezone=True), default=now_ist)


def ensure_columns(engine) -> None:
    """Add columns that postdate a table. create_all never alters an existing
    table, so without this a DB created before the column existed would fail
    every read of ChatSession. Idempotent; run by both the API and the worker."""
    from sqlalchemy import text
    with engine.begin() as c:
        c.execute(text("ALTER TABLE drs_chat_sessions ADD COLUMN IF NOT EXISTS review JSONB"))
        c.execute(text(
            "UPDATE drs_chat_sessions SET review = review || jsonb_build_object("
            "'run_id', md5(random()::text)) "
            "WHERE review->>'status' = 'done' AND review->>'run_id' IS NULL"))


class ReviewDecision(Base):
    """An officer's call on one review flag: 'accepted' (explained, with a
    note) or 'confirmed' (a real issue). Append-only — never updated or
    deleted — so the file keeps who decided what and when; the latest row per
    flag is the current state. `run_id` ties it to one review: a re-upload
    builds a new review, so old decisions stay in history but stop applying."""

    __tablename__ = "drs_review_decisions"

    id = Column(Integer, primary_key=True)
    chat_id = Column(String, index=True, nullable=False)
    run_id = Column(String, nullable=False)
    check_key = Column(String, nullable=False)
    check_label = Column(String)
    decision = Column(String, nullable=False)
    note = Column(Text)
    user_id = Column(Integer, nullable=False)
    username = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=now_ist)


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
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)

    sources = Column(JSONB)

    created_at = Column(DateTime(timezone=True), default=now_ist)


class IngestJob(Base):
    """One queued ingestion. The queue is this table, claimed by conditional
    UPDATE — no broker, because Neon is already here and a second service is
    not (upgrade_roadmap.txt PART 4 has the Redis line)."""

    __tablename__ = "drs_ingest_jobs"

    id = Column(String, primary_key=True)

    idempotency_key = Column(String, unique=True, index=True)

    chat_id = Column(String, index=True, nullable=False)
    user_id = Column(Integer, index=True, nullable=False)

    request_id = Column(String, index=True)

    filename = Column(String, nullable=False)
    payload = Column(LargeBinary, nullable=False)

    status = Column(String, default="queued", index=True, nullable=False)
    attempts = Column(Integer, default=0, nullable=False)
    max_attempts = Column(Integer, default=3, nullable=False)
    error = Column(Text)

    claimed_by = Column(String)
    claimed_at = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), default=now_ist)
    updated_at = Column(DateTime(timezone=True), default=now_ist)


Index("ix_drs_messages_chat_id_id", ChatMessage.chat_id, ChatMessage.id)
Index("ix_drs_sessions_user_updated", ChatSession.user_id, ChatSession.updated_at)
Index("ix_drs_jobs_status_created", IngestJob.status, IngestJob.created_at)
