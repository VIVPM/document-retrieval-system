"""Ingest worker. Claims jobs from drs_ingest_jobs and runs them."""

import asyncio
import os
import signal
import socket
import sys
import tempfile
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import job_queue
import logging_setup
from core.document_store import EnhancedDocumentStoreHybrid
from core.review import build_review
from db.database import SessionLocal, engine
from db.models import Account, ChatSession, ensure_columns, now_ist
from llm.llm_router import BREAKER_COOLDOWN_S, embed_model, llm, provider_down

logging_setup.configure()
log = logging_setup.get_logger("drs.worker")

MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
POLL_SECONDS = float(os.getenv("WORKER_POLL_SECONDS", "2"))
RECLAIM_EVERY = float(os.getenv("WORKER_RECLAIM_SECONDS", "60"))

INGEST_TIMEOUT_S = 900
if INGEST_TIMEOUT_S >= job_queue.LEASE_SECONDS:
    raise RuntimeError(
        f"INGEST_TIMEOUT_S ({INGEST_TIMEOUT_S}) must be below "
        f"INGEST_LEASE_SECONDS ({job_queue.LEASE_SECONDS}), or a job's lease "
        "expires while it is still running and a second worker claims it."
    )

WORKER_ID = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"

_shutdown = asyncio.Event()


def _ns(db, user_id: int) -> str:
    """Pinecone namespace for an account: the account's username."""
    acct = db.query(Account).filter(Account.id == user_id).first()
    if acct is None:
        raise RuntimeError(f"No account for user_id={user_id}")
    return acct.username


def _set_stage(chat_id: str, key: str) -> None:
    """Write the current sub-step so /status can surface it. Own session, and
    swallows everything -- a progress write must never break an ingest."""
    try:
        db = SessionLocal()
        try:
            db.query(ChatSession).filter(ChatSession.id == chat_id).update(
                {ChatSession.stage: key})
            db.commit()
        finally:
            db.close()
    except Exception:
        pass


def run_job(job: dict) -> None:
    """Ingest one document. Blocking; called in a thread."""
    chat_id, filename = job["chat_id"], job["filename"]
    last_attempt = job["attempts"] >= job["max_attempts"]
    deadline = time.monotonic() + INGEST_TIMEOUT_S

    def stage(key: str) -> None:
        """Record progress, and abort if the job has run out of wall clock."""
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Ingest exceeded INGEST_TIMEOUT_S ({INGEST_TIMEOUT_S}s) at stage {key!r}.")
        _set_stage(chat_id, key)

    tmp_path = None
    db = SessionLocal()
    try:
        chat = db.query(ChatSession).filter(ChatSession.id == chat_id).first()
        if chat is None:
            log.warning("chat gone, dropping job",
                        extra={"job_id": job["id"], "chat_id": chat_id})
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            tmp.write(job["payload"])

        store = EnhancedDocumentStoreHybrid(namespace=_ns(db, chat.user_id),
                                            chat_id=chat_id)
        success, stats = store.process_pdf(
            tmp_path, filename=filename, embed_model=embed_model,
            on_stage=stage,
        )

        db.refresh(chat)
        chat.stage = None
        if success:
            chat.status = "ready"
            chat.error = None
            chat.doc_stats = _sanitize(stats)
            chat.bm25_params = store.export_bm25_params()
            chat.review = {"status": "running"}
            if chat.title == "New Chat":
                chat.title = filename[:120]
            chat.updated_at = now_ist()
            db.commit()
            log.info("chat ready", extra={"job_id": job["id"], "chat_id": chat_id,
                                          "chunks": (stats or {}).get("total_chunks")})
            _run_review(chat_id, store.logical_docs)
            return

        store.retriever.delete_chat()
        reason = str(stats.get("error", "Processing failed."))[:2000]
        if last_attempt:
            chat.status = "failed"
            chat.error = reason
            chat.updated_at = now_ist()
        db.commit()
        raise RuntimeError(reason)

    except Exception as e:
        db.rollback()
        if last_attempt:
            try:
                chat = db.query(ChatSession).filter(ChatSession.id == chat_id).first()
                if chat is not None and chat.status == "processing":
                    chat.status = "failed"
                    chat.stage = None
                    chat.error = f"{type(e).__name__}: {e}"[:2000]
                    chat.updated_at = now_ist()
                    db.commit()
            except Exception:
                db.rollback()
        raise
    finally:
        db.close()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _run_review(chat_id: str, logical_docs) -> None:
    """Build the automatic file review and store it on the chat."""
    started = time.monotonic()
    try:
        review = build_review(logical_docs)
        review["run_id"] = uuid.uuid4().hex
    except Exception as e:
        log.exception("review failed", extra={"chat_id": chat_id})
        review = {"status": "failed", "error": f"{type(e).__name__}: {e}"[:500]}
    db = SessionLocal()
    try:
        chat = db.query(ChatSession).filter(ChatSession.id == chat_id).first()
        if chat is not None:
            chat.review = _sanitize(review)
            db.commit()
        log.info("review stored", extra={"chat_id": chat_id, "status": review.get("status"),
                                         "summary": review.get("summary"),
                                         "duration_s": round(time.monotonic() - started, 1)})
    except Exception:
        db.rollback()
        log.exception("review store failed", extra={"chat_id": chat_id})
    finally:
        db.close()


def _sanitize(obj):
    """JSONB-safe copy: numpy scalars and NaN are not valid JSON."""
    import math
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "item"):
        try:
            return _sanitize(obj.item())
        except Exception:
            return str(obj)
    return obj


async def _reclaim_loop() -> None:
    """Return jobs from workers that died to the queue, forever."""
    while not _shutdown.is_set():
        try:
            n = job_queue.reclaim_stale()
            if n:
                log.warning("reclaimed stale jobs", extra={"count": n})
        except Exception:
            log.exception("reclaim failed")
        try:
            await asyncio.wait_for(_shutdown.wait(), timeout=RECLAIM_EVERY)
        except asyncio.TimeoutError:
            pass


async def _run_one(job: dict, sem: asyncio.Semaphore) -> None:
    """Run a claimed job in a thread and record its outcome."""
    async with sem:
        jid = job["id"]
        logging_setup.set_correlation_id(job.get("request_id") or jid[:16])
        started = time.monotonic()
        log.info("ingest started", extra={"job_id": jid, "chat_id": job["chat_id"],
                                          "doc_filename": job["filename"],
                                          "attempt": job["attempts"],
                                          "max_attempts": job["max_attempts"]})
        try:
            await asyncio.to_thread(run_job, job)
            job_queue.finish(job["id"], ok=True)
            log.info("ingest finished", extra={
                "job_id": jid, "chat_id": job["chat_id"],
                "duration_s": round(time.monotonic() - started, 1)})
        except Exception as e:
            will_retry = job["attempts"] < job["max_attempts"]
            log.error("ingest failed", exc_info=True, extra={
                "job_id": jid, "chat_id": job["chat_id"],
                "attempt": job["attempts"], "will_retry": will_retry,
                "duration_s": round(time.monotonic() - started, 1)})
            try:
                job_queue.finish(job["id"], ok=False, error=f"{type(e).__name__}: {e}")
            except Exception:
                log.exception("could not record job failure", extra={"job_id": jid})


async def main() -> None:
    await asyncio.to_thread(ensure_columns, engine)
    log.info("worker up", extra={
        "worker_id": WORKER_ID, "concurrency": MAX_CONCURRENT_JOBS,
        "poll_s": POLL_SECONDS, "job_timeout_s": INGEST_TIMEOUT_S,
        "lease_s": job_queue.LEASE_SECONDS, "llm": llm.label})

    sem = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    running: set[asyncio.Task] = set()
    reclaimer = asyncio.create_task(_reclaim_loop())

    while not _shutdown.is_set():
        if len(running) >= MAX_CONCURRENT_JOBS:
            await asyncio.sleep(0.2)
            running = {t for t in running if not t.done()}
            continue

        if provider_down():
            log.warning("provider down, pausing claims",
                        extra={"cooldown_s": BREAKER_COOLDOWN_S})
            try:
                await asyncio.wait_for(_shutdown.wait(), timeout=BREAKER_COOLDOWN_S)
            except asyncio.TimeoutError:
                pass
            continue

        try:
            job = await asyncio.to_thread(job_queue.claim, WORKER_ID)
        except Exception:
            log.exception("claim failed")
            job = None

        if job is None:
            try:
                await asyncio.wait_for(_shutdown.wait(), timeout=POLL_SECONDS)
            except asyncio.TimeoutError:
                pass
            running = {t for t in running if not t.done()}
            continue

        task = asyncio.create_task(_run_one(job, sem))
        running.add(task)
        task.add_done_callback(running.discard)

    if running:
        log.info("draining before shutdown", extra={"in_flight": len(running)})
        await asyncio.gather(*running, return_exceptions=True)
    reclaimer.cancel()
    log.info("worker stopped cleanly")


def _handle_signal(signum, _frame) -> None:
    print(f"Signal {signum} received - finishing in-flight jobs, then exiting.",
          flush=True)
    try:
        asyncio.get_running_loop().call_soon_threadsafe(_shutdown.set)
    except RuntimeError:
        _shutdown.set()


if __name__ == "__main__":
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, AttributeError, OSError):
            pass
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
