"""
The ingest queue: enqueue, claim, finish, reclaim.

Postgres IS the broker. `drs_ingest_jobs` is the queue and a conditional
UPDATE is the claim, so there is no Redis and no second service to run
(upgrade_roadmap.txt PART 4 holds the Redis line, for when there is more than
one instance).

The claim is the only part that has to be exactly right. Two workers polling
the same table will both see the same oldest queued row, so the SELECT cannot
decide who gets it -- the UPDATE must, by matching on the status it expects to
find. Whoever's UPDATE reports a row is the owner; the loser sees 0 rows and
moves on. `FOR UPDATE SKIP LOCKED` does the same job in one statement and is
what makes a second worker safe to start.
"""

import os
import uuid
from datetime import timedelta

from sqlalchemy import text

from db.database import SessionLocal
from db.models import ChatSession, IngestJob, now_ist

# How long a claim is good for. A worker that dies mid-ingest cannot release
# its own job, so this is what lets another one take it. It must exceed the
# slowest realistic ingest (a 50-page packet is minutes of Textract), or a
# healthy worker's job gets stolen while it is still running -- which is worse
# than the crash it protects against, because then two workers write the same
# chat.
LEASE_SECONDS = int(os.getenv("INGEST_LEASE_SECONDS", "1800"))

MAX_ATTEMPTS = int(os.getenv("INGEST_MAX_ATTEMPTS", "3"))


def enqueue(db, chat_id: str, user_id: int, filename: str, payload: bytes) -> str:
    """Add one ingestion to the queue. Caller owns the transaction.

    Does not commit: the API enqueues in the same transaction that flips the
    chat to 'processing', so a job can never exist for a chat that was never
    marked, nor the reverse.
    """
    job = IngestJob(
        id=uuid.uuid4().hex,
        chat_id=chat_id,
        user_id=user_id,
        filename=filename,
        payload=payload,
        status="queued",
        attempts=0,
        max_attempts=MAX_ATTEMPTS,
    )
    db.add(job)
    return job.id


def claim(worker_id: str):
    """Take the oldest queued job, or None. Safe against other workers.

    SKIP LOCKED is what makes it safe: the row is locked inside the same
    statement that updates it, and a second worker's identical query passes
    over the locked row instead of blocking on it or claiming it twice.

    Returns a detached snapshot (id, chat_id, user_id, filename, payload), not
    an ORM object -- the session closes here, and the worker holds this while
    it ingests for minutes. Keeping a live session open for that long would
    pin a pooled connection for the whole job.
    """
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
                    ORDER BY created_at
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
             )
         RETURNING id, chat_id, user_id, filename, payload, attempts, max_attempts
        """), {"worker": worker_id, "now": now_ist()}).mappings().first()
        db.commit()
        return dict(row) if row else None
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def finish(job_id: str, ok: bool, error: str | None = None) -> None:
    """Mark a claimed job done or failed, releasing its lease either way.

    A failure with attempts left goes back to 'queued' rather than 'failed', so
    the reclaim sweep is not the only path back. Retrying here is safe because
    ingest is idempotent by teardown: a failed run deletes its own half-written
    vectors before the status is written.
    """
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
        job.claimed_by = None
        job.claimed_at = None
        job.updated_at = now_ist()
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def reclaim_stale() -> int:
    """Return jobs whose lease expired to the queue; fail those out of attempts.

    This is what replaces the old startup reaper. The reaper flipped every
    'processing' chat to 'failed' on the assumption that the only worker was
    this process, so a restart meant the work was gone. With a queue the work
    is not gone -- it is a row -- so the right move is to re-run it, and the
    only jobs that should fail are those that have already burned their
    attempts.

    Runs on a timer in the worker, not just at startup, because a worker can
    die at any point and nothing else notices.
    """
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
            job.claimed_by = None
            job.claimed_at = None
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
    """Move a chat out of 'processing' when its job will never run again.

    A chat left on 'processing' polls forever and cannot be deleted (the delete
    endpoint guards against removing a chat whose ingest is in flight), so a
    terminally failed job has to say so on the chat too.
    """
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
