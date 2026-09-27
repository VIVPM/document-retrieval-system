"""The ingest queue: enqueue, claim, finish, reclaim."""

import hashlib
import os
import uuid
from datetime import timedelta

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from db.database import SessionLocal
from db.models import ChatSession, IngestJob, now_ist

LEASE_SECONDS = 1800

MAX_ATTEMPTS = int(os.getenv("INGEST_MAX_ATTEMPTS", "3"))


def content_key(chat_id: str, payload: bytes) -> str:
    """Idempotency key for an upload: the chat plus a hash of the bytes."""
    return f"{chat_id}:{hashlib.sha256(payload).hexdigest()[:32]}"


def enqueue(db, chat_id: str, user_id: int, filename: str, payload: bytes,
            idempotency_key: str | None = None,
            request_id: str | None = None) -> tuple[str, bool]:
    """Add one ingestion to the queue. Returns (job_id, created)."""
    key = idempotency_key or content_key(chat_id, payload)

    existing = db.query(IngestJob).filter(IngestJob.idempotency_key == key).first()
    if existing is not None:
        return existing.id, False

    job = IngestJob(
        id=uuid.uuid4().hex,
        idempotency_key=key,
        request_id=request_id,
        chat_id=chat_id,
        user_id=user_id,
        filename=filename,
        payload=payload,
        status="queued",
        attempts=0,
        max_attempts=MAX_ATTEMPTS,
    )
    try:
        with db.begin_nested():
            db.add(job)
            db.flush()
        return job.id, True
    except IntegrityError:
        loser = db.query(IngestJob).filter(IngestJob.idempotency_key == key).first()
        return (loser.id if loser else job.id), False


def claim(worker_id: str):
    """Take the oldest queued job, or None. Safe against other workers."""
    db = SessionLocal()
    try:
        row = db.execute(text("""
            UPDATE drs_ingest_jobs
               SET status = 'running',
                   attempts = attempts + 1,
                   claimed_by = :worker,
                   claimed_at = :now,
                   updated_at = :now
             WHERE id = (
                   SELECT id FROM drs_ingest_jobs
                    WHERE status = 'queued'
                    ORDER BY (
                        -- When this user was last served, across ALL their
                        -- jobs. NULL means never, and NULLS FIRST puts a new
                        -- user at the head -- so every user's turn comes round
                        -- before anyone gets a second one.
                        --
                        -- Counting a user's QUEUED jobs was tried first and is
                        -- wrong: the count collapses as their jobs leave the
                        -- queue, so after the batch owner's first job is
                        -- claimed their second ranks 0 again and wins on age.
                        -- Measured: it produced AAAAAAAABC, i.e. strict FIFO.
                        SELECT MAX(peer.claimed_at) FROM drs_ingest_jobs peer
                         WHERE peer.user_id = drs_ingest_jobs.user_id
                    ) NULLS FIRST, created_at
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
             )
         RETURNING id, chat_id, user_id, filename, payload, attempts,
                   max_attempts, request_id
        """), {"worker": worker_id, "now": now_ist()}).mappings().first()
        db.commit()
        if row is None:
            return None
        job = dict(row)
        job["payload"] = bytes(job["payload"])
        return job
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def finish(job_id: str, ok: bool, error: str | None = None) -> None:
    """Mark a claimed job done or failed, releasing its lease either way."""
    db = SessionLocal()
    try:
        job = db.query(IngestJob).filter(IngestJob.id == job_id).first()
        if job is None:
            return
        if ok:
            job.status = "done"
            job.error = None
        elif job.attempts < job.max_attempts:
            job.status = "queued"
            job.error = (error or "")[:2000]
        else:
            job.status = "failed"
            job.error = (error or "")[:2000]

        if job.status in ("done", "failed"):
            job.payload = b""

        job.updated_at = now_ist()
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def reclaim_stale() -> int:
    """Return jobs whose lease expired to the queue; fail those out of attempts."""
    deadline = now_ist() - timedelta(seconds=LEASE_SECONDS)
    db = SessionLocal()
    try:
        stale = db.query(IngestJob).filter(
            IngestJob.status == "running",
            IngestJob.claimed_at < deadline,
        ).all()
        for job in stale:
            if job.attempts < job.max_attempts:
                job.status = "queued"
                job.error = "Worker stopped responding; requeued."
            else:
                job.status = "failed"
                job.error = "Worker stopped responding and no attempts remain."
                _fail_chat(db, job.chat_id, job.error)
            job.updated_at = now_ist()
        if stale:
            db.commit()
        return len(stale)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _fail_chat(db, chat_id: str, reason: str) -> None:
    """Move a chat out of 'processing' when its job will never run again."""
    chat = db.query(ChatSession).filter(ChatSession.id == chat_id).first()
    if chat is not None and chat.status == "processing":
        chat.status = "failed"
        chat.stage = None
        chat.error = reason[:2000]
        chat.updated_at = now_ist()


def depth() -> dict:
    """Queue depth by status. The number autoscaling would read (PART 4)."""
    db = SessionLocal()
    try:
        rows = db.execute(text(
            "SELECT status, COUNT(*) AS n FROM drs_ingest_jobs GROUP BY status"
        )).mappings().all()
        return {r["status"]: r["n"] for r in rows}
    finally:
        db.close()
