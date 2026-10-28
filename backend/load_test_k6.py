"""Local k6 load test: step reviewers through the real API with stub ingest workers.

    python load_test_k6.py --render-suspended --levels 25,50,75,100,150

The API and workers run on this machine against the database in backend/.env,
so suspend Render (and any other worker on that queue) first. Workers replace
ingestion with a fixed wait: no LLM, embedding or Pinecone calls are made.
Test accounts and everything they created are deleted when the run ends.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402

from auth import hash_password  # noqa: E402
from db.database import SessionLocal, engine  # noqa: E402
from db.models import Account  # noqa: E402

OUT = ROOT / "k6_results"
SCRIPT = ROOT / "load_test_k6.js"
PASSWORD = "k6-load-pw-7315"
ENDPOINTS = ["login", "health", "chats", "chat", "create_chat", "upload", "job_status"]
# Ingestion never parses this file: the stub worker only waits, then marks the chat ready.
PDF = (b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
       b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
       b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
       b"trailer<</Root 1 0 R>>\n%%EOF\n")

STUB_WORKER = r"""
import asyncio, os, time
import worker
from db.database import SessionLocal
from db.models import ChatSession, now_ist

def stub_run_job(job):
    time.sleep(float(os.environ["K6_STUB_SECONDS"]))
    db = SessionLocal()
    try:
        chat = db.get(ChatSession, job["chat_id"])
        if chat is not None:
            chat.status, chat.stage, chat.error = "ready", None, None
            chat.doc_stats = {"pages": 1, "load_test_stub": True}
            chat.updated_at = now_ist()
            db.commit()
    finally:
        db.close()

worker.run_job = stub_run_job
asyncio.run(worker.main())
"""


def preflight() -> None:
    """A stub worker must never claim a real user's job on the shared queue."""
    with engine.connect() as db:
        active = db.execute(text(
            "SELECT count(*) FROM drs_ingest_jobs WHERE status IN ('queued', 'running')")).scalar()
    if active:
        raise SystemExit(f"ABORT: {active} job(s) already queued/running; wait for the queue to drain.")


def create_accounts(prefix: str, count: int) -> None:
    hashed = hash_password(PASSWORD)
    db = SessionLocal()
    try:
        db.add_all(Account(username=f"{prefix}{i}", hashed_password=hashed)
                   for i in range(1, count + 1))
        db.commit()
    finally:
        db.close()


def cleanup(prefix: str) -> int:
    """Delete every row the test accounts produced, children first."""
    like = {"p": f"{prefix}%"}
    users = "SELECT id FROM drs_accounts WHERE username LIKE :p"
    with engine.begin() as db:
        n = db.execute(text("SELECT count(*) FROM drs_accounts WHERE username LIKE :p"), like).scalar() or 0
        db.execute(text(f"DELETE FROM drs_chat_documents WHERE chat_id IN "
                        f"(SELECT id FROM drs_chat_sessions WHERE user_id IN ({users}))"), like)
        for table in ("drs_ingest_jobs", "drs_chat_messages", "drs_review_decisions",
                      "drs_chat_sessions", "drs_refresh_tokens", "drs_account_providers"):
            db.execute(text(f"DELETE FROM {table} WHERE user_id IN ({users})"), like)
        db.execute(text("DELETE FROM drs_login_failures WHERE username LIKE :p"), like)
        db.execute(text("DELETE FROM drs_accounts WHERE username LIKE :p"), like)
    return n


def job_stats(prefix: str, since: datetime) -> dict:
    with engine.connect() as db:
        rows = db.execute(text(
            "SELECT j.status, EXTRACT(EPOCH FROM (j.claimed_at - j.created_at)) "
            "FROM drs_ingest_jobs j JOIN drs_accounts a ON a.id = j.user_id "
            "WHERE a.username LIKE :p AND j.created_at >= :since"),
            {"p": f"{prefix}%", "since": since}).all()
    waits = sorted(float(w) for _, w in rows if w is not None)

    def pick(q):
        return round(waits[min(len(waits) - 1, int(q * len(waits)))], 1) if waits else None

    return {"submitted": len(rows),
            "done": sum(s == "done" for s, _ in rows),
            "failed": sum(s == "failed" for s, _ in rows),
            "unfinished": sum(s in ("queued", "running") for s, _ in rows),
            "queue_wait_s": {"p50": pick(.50), "p95": pick(.95), "p99": pick(.99)}}


def wait_for_drain(prefix: str, since: datetime, timeout: float) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end and job_stats(prefix, since)["unfinished"]:
        time.sleep(3)


def wait_for_health(base: str, timeout: float = 120) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if urlopen(f"{base}/api/health", timeout=5).status == 200:
                return
        except OSError:
            time.sleep(1)
    raise SystemExit("API did not become healthy.")


def spawn(cmd: list[str], env: dict, log_path: Path) -> subprocess.Popen:
    log = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def run_level(users: int, args, base: str, prefix: str, pdf_path: Path, tag: str) -> dict:
    stages = [{"duration": f"{args.ramp_up}s", "target": users},
              {"duration": f"{args.hold}s", "target": users},
              {"duration": "10s", "target": 0}]
    summary_path = OUT / f"{tag}_k6_{users}.json"
    env = {**os.environ, "BASE": base, "PREFIX": prefix, "PASSWORD": PASSWORD,
           "PDF": str(pdf_path), "STAGES": json.dumps(stages), "OUT": str(summary_path)}
    since = datetime.now().astimezone()
    print(f"\n--- {users} users: {args.ramp_up}s ramp, {args.hold}s hold")
    subprocess.run([args.k6, "run", "--quiet", str(SCRIPT)], cwd=ROOT, env=env, check=False)
    wait_for_drain(prefix, since, args.drain_timeout)

    metrics = json.loads(summary_path.read_text(encoding="utf-8"))["metrics"]
    summary_path.unlink()

    def trend(m):
        return {k: round(v) for k, v in metrics.get(m, {}).get("values", {}).items()
                if k in ("p(50)", "p(95)", "p(99)", "count")}

    endpoints = {name: {**trend(f"api_{name}"),
                        "error_pct": round(100 * metrics.get(f"api_{name}_failed", {})
                                           .get("values", {}).get("rate", 0), 2)}
                 for name in ENDPOINTS}
    failed = sorted(m for m, v in metrics.items()
                    if any(not t["ok"] for t in v.get("thresholds", {}).values()))
    jobs = job_stats(prefix, since)
    error_pct = round(100 * metrics["http_req_failed"]["values"]["rate"], 2)
    row = {"users": users, "requests": metrics["http_reqs"]["values"]["count"],
           "error_pct": error_pct, "thresholds_failed": failed,
           "passed": not failed and jobs["done"] == jobs["submitted"],
           "endpoints": endpoints, "job_total_ms": trend("job_total"), "jobs": jobs}
    print(f"    requests {row['requests']}  errors {error_pct}%  "
          f"jobs {jobs['done']}/{jobs['submitted']}  failed thresholds: {failed or 'none'}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Local k6 load test with stub ingest workers.")
    ap.add_argument("--render-suspended", action="store_true",
                    help="confirm no Render or other worker is claiming jobs from this database")
    ap.add_argument("--levels", default="25,50,75,100,150")
    ap.add_argument("--ramp-up", type=int, default=30)
    ap.add_argument("--hold", type=int, default=180)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--slots", type=int, default=20, help="concurrent jobs per worker")
    ap.add_argument("--stub-seconds", type=float, default=4.0)
    ap.add_argument("--drain-timeout", type=float, default=600)
    ap.add_argument("--stop-error-pct", type=float, default=25)
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--k6", default=shutil.which("k6") or r"C:\Program Files\k6\k6.exe")
    args = ap.parse_args()
    if not args.render_suspended:
        ap.error("pass --render-suspended once no other worker is using this database")
    levels = [int(x) for x in args.levels.split(",")]

    preflight()
    OUT.mkdir(exist_ok=True)
    tag = f"k6-{uuid.uuid4().hex[:8]}"
    prefix = f"{tag.replace('-', '_')}_u"
    base = f"http://127.0.0.1:{args.port}"
    pdf_path = OUT / f"{tag}.pdf"
    pdf_path.write_bytes(PDF)
    env = {**os.environ, "PYTHONUTF8": "1", "RUN_WORKER_IN_PROCESS": "0",
           "RATE_LOGIN": "100000/minute", "RATE_UPLOAD": "100000/hour",
           "LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": "",
           "MAX_CONCURRENT_JOBS": str(args.slots), "K6_STUB_SECONDS": str(args.stub_seconds)}
    procs, rows = [], []
    try:
        create_accounts(prefix, max(levels))
        procs.append(spawn([sys.executable, "-m", "uvicorn", "main:app", "--port", str(args.port),
                            "--log-level", "warning"], env, OUT / f"{tag}_server.log"))
        procs += [spawn([sys.executable, "-c", STUB_WORKER], env, OUT / f"{tag}_worker_{i}.log")
                  for i in range(args.workers)]
        wait_for_health(base)
        for users in levels:
            rows.append(run_level(users, args, base, prefix, pdf_path, tag))
            if rows[-1]["error_pct"] > args.stop_error_pct:
                print(f"Stopping: errors above {args.stop_error_pct}%.")
                break
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=30)
        pdf_path.unlink(missing_ok=True)
        print(f"Cleaned up {cleanup(prefix)} test accounts.")

    passing = [r["users"] for r in rows if r["passed"]]
    result = {"settings": {**{k: v for k, v in vars(args).items() if k != "k6"}, "tag": tag},
              "levels": rows, "highest_passing_users": max(passing) if passing else None}
    path = OUT / f"k6_load_{datetime.now():%Y-%m-%d_%H-%M-%S}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nHighest passing level: {result['highest_passing_users']}  ->  {path}")


if __name__ == "__main__":
    main()
