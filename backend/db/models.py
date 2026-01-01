"""
Neon tables.

A chat owns exactly one document. Its vectors live in the owning user's
Pinecone namespace under ids prefixed with the chat id.

`drs_chat_sessions` is the chat itself, not a session layer on top of one —
the name predates the current model. It is the only home for bm25_params,
ownership and ingest status, so it cannot be derived from the messages table.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import Column, DateTime, Index, Integer, String, Text
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


# The common read is "all messages of one chat, in order".
Index("ix_drs_messages_chat_id_id", ChatMessage.chat_id, ChatMessage.id)
# The common list is "my chats, most recent first".
Index("ix_drs_sessions_user_updated", ChatSession.user_id, ChatSession.updated_at)
