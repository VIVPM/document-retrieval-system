-- Schema for the Document Retrieval System (Neon / Postgres).
-- Idempotent: safe to re-run. Paste into the Neon SQL editor, or:
--   python -c "from db.database import engine, Base; import db.models; Base.metadata.create_all(engine)"
--
-- Tables are prefixed drs_ so this schema can share a Neon database with other
-- projects without colliding on names like `accounts` or `chat_messages`.

CREATE TABLE IF NOT EXISTS drs_accounts (
    id              SERIAL PRIMARY KEY,
    username        TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_drs_accounts_username ON drs_accounts (username);


-- Login lockout state. In Postgres rather than process memory so the lockout
-- holds across API instances and survives a free-tier cold start.
CREATE TABLE IF NOT EXISTS drs_login_failures (
    id         SERIAL PRIMARY KEY,
    username   TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_drs_login_failures_username ON drs_login_failures (username);


-- Refresh tokens: opaque, long-lived, revocable. Stored as a SHA-256 hash so a
-- DB leak can't be replayed. Deleting a row revokes it (that is /logout).
CREATE TABLE IF NOT EXISTS drs_refresh_tokens (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    token_hash TEXT UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_drs_refresh_user ON drs_refresh_tokens (user_id);
CREATE INDEX IF NOT EXISTS ix_drs_refresh_hash ON drs_refresh_tokens (token_hash);


-- A chat, owning one document. Its vectors live in the user's namespace under
-- ids prefixed with this id. (Table name predates the current model.)
CREATE TABLE IF NOT EXISTS drs_chat_sessions (
    id          TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    title       TEXT DEFAULT 'New Chat',
    -- awaiting_document -> processing -> ready | failed
    status      TEXT DEFAULT 'awaiting_document',
    error       TEXT,
    filename    TEXT,
    doc_stats   JSONB,      -- pages, doc types, chunk count — renders the UI without rehydrating
    bm25_params JSONB,      -- fitted BM25Encoder.get_params(); restores the sparse half after a restart
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_drs_sessions_user_updated
    ON drs_chat_sessions (user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS ix_drs_sessions_status ON drs_chat_sessions (status);


CREATE TABLE IF NOT EXISTS drs_chat_messages (
    id         SERIAL PRIMARY KEY,
    chat_id    TEXT NOT NULL,
    user_id    INTEGER NOT NULL,
    role       TEXT NOT NULL,     -- 'user' | 'assistant'
    content    TEXT NOT NULL,
    sources    JSONB,             -- citations; an answer without provenance is unusable here
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_drs_messages_chat_id_id ON drs_chat_messages (chat_id, id);
CREATE INDEX IF NOT EXISTS ix_drs_messages_user_id ON drs_chat_messages (user_id);


-- Housekeeping, run manually or from a scheduler.
--
-- Stranded-ingest reaping now runs automatically at startup (main.py
-- _reap_stranded_ingests) — this UPDATE is the equivalent for a scheduler if
-- you ever run more than one worker and want time-based instead of on-restart.
--   UPDATE drs_chat_sessions
--      SET status = 'failed', error = 'Processing was interrupted. Please re-upload.'
--    WHERE status = 'processing' AND updated_at < now() - INTERVAL '30 minutes';
--
-- Expired refresh tokens are dead weight (refresh already rejects them). Sweep:
--   DELETE FROM drs_refresh_tokens WHERE expires_at < now();
--
-- Find namespaces eligible for deletion (drop them from Pinecone FIRST, then
-- delete the rows — an orphaned namespace has no owner left to find it).
--   SELECT id FROM drs_chat_sessions WHERE updated_at < now() - INTERVAL '30 days';
