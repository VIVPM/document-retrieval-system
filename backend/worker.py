"""
Ingest worker. Claims jobs from drs_ingest_jobs and runs them.

Run it beside the API, not inside it:

    python -m worker                      # from backend/
    python backend/worker.py              # from the repo root

Why a separate process at all: ingestion used to run in a FastAPI
BackgroundTask, which lives and dies with the API process, so every deploy or
crash destroyed whatever was in flight and the user saw only a 'failed' badge.
A queue plus this loop means the work is a row that outlives any one process.

Concurrency is asyncio over a thread pool, not multiprocessing.
`process_pdf` is blocking and spends nearly all of its wall-clock time waiting
on Textract, Gemini and Pinecone, so threads are the right shape -- the GIL is
released for every one of those calls. MAX_CONCURRENT_JOBS caps how many run at
once; the real ceiling is provider rate limits, not CPU.
"""

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
from db.database import SessionLocal
from db.models import Account, ChatSession, now_ist
from llm.llm_router import BREAKER_COOLDOWN_S, embed_model, llm, provider_down

logging_setup.configure()
log = logging_setup.get_logger("drs.worker")

MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
POLL_SECONDS = float(os.getenv("WORKER_POLL_SECONDS", "2"))
RECLAIM_EVERY = float(os.getenv("WORKER_RECLAIM_SECONDS", "60"))

# Wall-clock ceiling on ONE ingest. Per-call timeouts in llm_router bound a
# single hung request; this bounds the job as a whole, which is a different
# failure -- a document that keeps making slow progress (many pages, each call
# succeeding just under its own timeout) would otherwise occupy a concurrency
# slot indefinitely.
#
# A constant, not an env var: it is half of an invariant with
# job_queue.LEASE_SECONDS and must stay BELOW it. If it were higher the lease
# would expire first, a second worker would claim a job this one is still
# running, and both would write the same chat. Env vars let the two drift
# apart in one environment and not another, which is exactly how a pair like
# this breaks silently. The assert below is the backstop, not the rule.
INGEST_TIMEOUT_S = 900
if INGEST_TIMEOUT_S >= job_queue.LEASE_SECONDS:
    raise RuntimeError(
        f"INGEST_TIMEOUT_S ({INGEST_TIMEOUT_S}) must be below "
        f"INGEST_LEASE_SECONDS ({job_queue.LEASE_SECONDS}), or a job's lease "
        "expires while it is still running and a second worker claims it."
    )

# hostname + a random suffix: the host tells you WHICH box a job ran on, the
# suffix keeps two workers on one box distinct. socket.gethostname works on
# Windows too, where os.uname does not exist at all.
WORKER_ID = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"

_shutdown = asyncio.Event()


def _ns(db, user_id: int) -> str:
    """Pinecone namespace for an account: the account's username.

    Duplicated from main rather than imported -- importing main would build the
    whole FastAPI app, its rate limiter and its startup hooks inside the
    worker. Kept in step with main._ns; if the namespace rule ever changes it
    must change in both (CLAUDE.md records the rule).
    """
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
    """Ingest one document. Blocking; called in a thread.

    Every exit path must leave the chat at 'ready' or 'failed'. A chat stuck on
    'processing' is indistinguishable from one still working, and the delete
    guard makes it undeletable.

    Raises on failure so the caller can decide whether the JOB retries. The
    chat is only marked failed once no attempts remain -- a chat flipped to
    'failed' while its job is still queued for retry would tell the user the
    work was lost while a worker is about to pick it up again.
    """
    chat_id, filename = job["chat_id"], job["filename"]
    last_attempt = job["attempts"] >= job["max_attempts"]
    deadline = time.monotonic() + INGEST_TIMEOUT_S

    def stage(key: str) -> None:
        """Record progress, and abort if the job has run out of wall clock.

        Cooperative on purpose. A thread cannot be killed from outside, so
        asyncio.wait_for would report a timeout and then still block until the
        thread finished -- freeing nothing. Raising from inside the thread is
        what actually ends it: process_pdf catches it and returns a clean
        failure, so teardown and requeue happen on the normal error path.

        Granularity is one stage. A hang INSIDE a stage is bounded instead by
        LLM_TIMEOUT_S on each provider call, which is the realistic case.
        """
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

        # The bytes live in the job row, so write them somewhere process_pdf
        # can open. Deleted in the finally, whatever happens.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            tmp.write(job["payload"])

        store = EnhancedDocumentStoreHybrid(namespace=_ns(db, chat.user_id),
                                            chat_id=chat_id)
        success, stats = store.process_pdf(
            tmp_path, filename=filename, embed_model=embed_model,
            on_stage=stage,
        )

        # _set_stage wrote `stage` on a separate session, so this one's copy is
        # stale. Refreshing before the terminal commit is what makes clearing
        # it stick.
        db.refresh(chat)
        chat.stage = None
        if success:
            chat.status = "ready"
            chat.error = None
            chat.doc_stats = _sanitize(stats)
            chat.bm25_params = store.export_bm25_params()
            if chat.title == "New Chat":
                chat.title = filename[:120]
            chat.updated_at = now_ist()
            db.commit()
            log.info("chat ready", extra={"job_id": job["id"], "chat_id": chat_id,
                                          "chunks": (stats or {}).get("total_chunks")})
            return

        # Half-built vectors would compete for top_k on every later query, so
        # drop them. This is also what makes a retry safe.
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


def _sanitize(obj):
    """JSONB-safe copy: numpy scalars and NaN are not valid JSON.

    Local rather than imported from main for the same reason as _ns.
    """
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
    """Return jobs from workers that died to the queue, forever.

    On a timer rather than only at startup: a worker can die at any moment and
    nothing else notices its lease going stale.
    """
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
        # Adopt the id the API logged this upload under. Without it the worker
        # lines are a separate island and nothing connects an upload to the
        # ingest it caused.
        logging_setup.set_correlation_id(job.get("request_id") or jid[:16])
        started = time.monotonic()
        # doc_filename, not filename: LogRecord already owns `filename` (the
        # source file) and stdlib logging raises KeyError on the collision.
        log.info("ingest started", extra={"job_id": jid, "chat_id": job["chat_id"],
                                          "doc_filename": job["filename"],
                                          "attempt": job["attempts"],
                                          "max_attempts": job["max_attempts"]})
        try:
            # No wait_for here: run_job enforces its own deadline from inside
            # the thread. Wrapping this in wait_for would report a timeout and
            # then block on the thread anyway, freeing no slot.
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
            # finish() decides retry vs fail from the attempt count.
            try:
                job_queue.finish(job["id"], ok=False, error=f"{type(e).__name__}: {e}")
            except Exception:
                log.exception("could not record job failure", extra={"job_id": jid})


async def main() -> None:
    log.info("worker up", extra={
        "worker_id": WORKER_ID, "concurrency": MAX_CONCURRENT_JOBS,
        "poll_s": POLL_SECONDS, "job_timeout_s": INGEST_TIMEOUT_S,
        "lease_s": job_queue.LEASE_SECONDS, "llm": llm.label})

    sem = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    running: set[asyncio.Task] = set()
    reclaimer = asyncio.create_task(_reclaim_loop())

    while not _shutdown.is_set():
        # Only claim what there is room to run. Claiming past the semaphore
        # would hold leases on jobs sitting in a queue inside this process,
        # where another worker cannot take them either.
        if len(running) >= MAX_CONCURRENT_JOBS:
            await asyncio.sleep(0.2)
            running = {t for t in running if not t.done()}
            continue

        # Do not claim while the provider is down. This is the half of the
        # breaker that only a queue can have: without it a backlog burns its
        # whole retry budget against an outage and a recoverable blip becomes a
        # pile of permanently failed documents. Jobs stay queued instead, and
        # are picked up when the circuit closes.
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

    # Graceful shutdown: stop claiming, let in-flight ingests finish. Killing
    # them here would strand their chats on 'processing' until a lease expired,
    # which is the exact failure the queue exists to remove.
    if running:
        log.info("draining before shutdown", extra={"in_flight": len(running)})
        await asyncio.gather(*running, return_exceptions=True)
    reclaimer.cancel()
    log.info("worker stopped cleanly")


def _handle_signal(signum, _frame) -> None:
    # print, not log, on purpose: logging takes a lock, and a signal arriving
    # while another thread holds it deadlocks the handler. Everything else in
    # this module logs; this one line stays a write.
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
            pass          # not all signals exist on Windows
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
