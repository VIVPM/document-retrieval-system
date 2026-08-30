"""Load test: does the API stay responsive while /message is saturated, and how
many concurrent users can it hold?

Two questions, two modes:
  default  idle phase vs saturated phase (N answers streaming) — does browsing
           degrade while /message is flat out? Read the RATIO, not absolute ms.
  --ramp   step browse concurrency up until errors or latency break — the
           machine's own read-path ceiling (sync `def` endpoints run in anyio's
           40-thread pool over a 5+10 DB pool, so the knee is there, not the loop).

The retrieval + LLM boundary is stubbed, so a run is free and takes seconds.
What stays real: the async endpoints, the SQLAlchemy pool, JWT auth, SSE, and
every DB round trip a message or a browse makes.

    python backend/load_test.py --messages 15 --concurrency 30
    python backend/load_test.py --ramp                     # local capacity knee
    python backend/load_test.py --ramp --base <url>        # a live server
    python backend/load_test.py --seed-messages 40         # realistic GET payload
    python backend/load_test.py --calibrate 3              # real messages; costs money
    python backend/load_test.py --cleanup                  # after a crashed run
"""
import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

import httpx

# Signup requires a Gmail address (main.py GMAIL_RE), so the load user is one.
LOAD_USER = "loadtest@gmail.com"
LOAD_PASS = "LoadTest-pw-9137"
Q = "What is the total loan amount?"
PDF = os.path.join(os.path.dirname(BASE_DIR), "Test Blob File.pdf")


def pctl(xs, p):
    """Nearest-rank percentile (small n — interpolation would invent precision)."""
    if not xs:
        return float("nan")
    xs = sorted(xs)
    rank = max(1, math.ceil(p / 100 * len(xs)))
    return xs[min(rank, len(xs)) - 1]


def _num(x):
    """NaN is not valid JSON — an empty latency bucket serialises to null, not NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(x, 1)


# --- Serve mode — the real FastAPI app with retrieval + LLM stubbed ---

class _StubStore:
    """Stands in for the rehydrated retriever so a load run needs no Pinecone,
    no embeddings, and no ingested document."""
    def __init__(self):
        self.alpha = 0.5

    def set_alpha(self, a):
        self.alpha = a

    def retrieve_only(self, question, filter_type=None, k=6):
        return []


def serve_mode(port, msg_seconds):
    """Run the real app, stubbing only the retrieval + LLM boundary of the
    message path. main imports these names, and send_message calls them by
    those names, so patching them on `main` is what the request path picks up.

    Rate limits and the daily message cap are lifted here so synthetic load
    measures the app's capacity rather than the limiter (which is exercised on
    its own elsewhere). The cap matters as much as the limiters: --seed-messages
    writes user turns straight to the DB and _credits_used_today counts them, so
    a seeded run would exhaust a real cap before the message phase even starts.
    Set before `import main`, which reads the cap at module level."""
    os.environ["RATE_MESSAGE"] = "1000000/minute"
    os.environ["RATE_UPLOAD"] = "1000000/hour"
    os.environ["RATE_LOGIN"] = "1000000/minute"
    os.environ["RATE_SIGNUP"] = "1000/hour"
    os.environ["DAILY_MESSAGE_CAP"] = "100000000"

    # Keep synthetic load out of Langfuse/Grafana. Set empty, don't pop: main.py's
    # load_dotenv(override=False) would otherwise repopulate them from .env.
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
              "GRAFANA_OTLP_ENDPOINT", "GRAFANA_OTLP_AUTH"):
        os.environ[k] = ""

    import main

    # Ingestion moved out of the API into worker.py, so the spawned server no
    # longer finishes an upload on its own -- --calibrate would poll a chat
    # that stays 'processing' until its 900s timeout. Run the worker loop in a
    # daemon thread so a load-test server stays self-contained. Only
    # --calibrate ingests; the other phases never enqueue, so this thread just
    # polls an empty queue.
    import asyncio as _asyncio
    import threading as _threading

    def _worker_thread():
        import worker
        try:
            _asyncio.run(worker.main())
        except Exception as e:
            print(f"load-test worker stopped: {type(e).__name__}: {e}")

    _threading.Thread(target=_worker_thread, daemon=True, name="ingest-worker").start()

    def stub_stream(*_a, **_k):
        # Sync generator: send_message pumps it through asyncio.to_thread, so a
        # real sleep here simulates generation latency without blocking the loop.
        time.sleep(msg_seconds)
        for tok in ("The ", "Total ", "Loan ", "Amount ", "is ", "$380,000."):
            yield tok

    main.rewrite_standalone = lambda q, hist: q
    main._get_retriever = lambda db, chat: _StubStore()
    main.stream_answer = stub_stream

    import uvicorn
    uvicorn.run(main.app, host="127.0.0.1", port=port, log_level="error")


# --- Browse client ---

async def _hammer(base, token, chat_id, concurrency, duration, mix="read"):
    """Fire a browse mix and record every latency, returning real elapsed too.

    mix="read"  the realistic steady state — a browser open on the app: health,
                the sidebar list, and one conversation's messages. This answers
                "how many people can use the app at once".
    mix="all"   also logs in every cycle. Deliberately harsh (a real user logs
                in once), and login is rate-limited so its 429s are throttle, not
                error — kept for observing the limiter under load.
    """
    lat = {"health": [], "chats": [], "chat": [], "login": []}
    errors = {"count": 0, "samples": []}
    throttled = 0
    stop = time.monotonic() + duration
    auth = {"Authorization": f"Bearer {token}"}
    # httpx caps at max_connections=100 by default — below that a big ramp would
    # measure the client's own pool, not the server. Scale it to concurrency.
    limits = httpx.Limits(max_connections=concurrency + 20,
                          max_keepalive_connections=concurrency)

    cycle = [
        ("health", "GET", "/api/health", {}),
        ("chats", "GET", "/api/chats", {"headers": auth}),
        ("chat", "GET", f"/api/chats/{chat_id}", {"headers": auth}),
    ]
    if mix == "all":
        cycle.append(("login", "POST", "/api/auth/login",
                      {"json": {"username": LOAD_USER, "password": LOAD_PASS}}))

    async def one(client):
        nonlocal throttled
        while time.monotonic() < stop:
            for name, method, url, kwargs in cycle:
                t = time.monotonic()
                try:
                    r = await client.request(method, url, timeout=30, **kwargs)
                    dt = (time.monotonic() - t) * 1000
                    if r.status_code == 429:
                        throttled += 1
                    elif r.status_code >= 400:
                        errors["count"] += 1
                        if len(errors["samples"]) < 5:
                            errors["samples"].append(f"{name} {r.status_code} {r.text[:60]}")
                    else:
                        lat[name].append(dt)
                except Exception as exc:
                    errors["count"] += 1
                    if len(errors["samples"]) < 5:
                        errors["samples"].append(f"{name} {type(exc).__name__}")
                if time.monotonic() >= stop:
                    return

    started = time.monotonic()
    async with httpx.AsyncClient(base_url=base, limits=limits) as client:
        await asyncio.gather(*[one(client) for _ in range(concurrency)])
    # Real elapsed, not nominal duration: a slow in-flight request can run past
    # the stop time, so dividing by `duration` would overstate throughput.
    return {"lat": lat, "errors": errors, "throttled": throttled,
            "elapsed": time.monotonic() - started}


async def _message_load(base, token, chat_id, n_streamers, stop_evt):
    """N clients continuously POST /message and drain the SSE stream."""
    auth = {"Authorization": f"Bearer {token}"}
    sent = 0
    limits = httpx.Limits(max_connections=n_streamers + 10)

    async def loop(client):
        nonlocal sent
        while not stop_evt.is_set():
            try:
                async with client.stream(
                    "POST", f"/api/chats/{chat_id}/message",
                    json={"question": Q, "num_chunks": 6, "alpha": 0.5},
                    headers=auth, timeout=60,
                ) as r:
                    async for _ in r.aiter_lines():
                        pass
                sent += 1
            except Exception:
                pass

    async with httpx.AsyncClient(base_url=base, limits=limits) as client:
        await asyncio.gather(*[loop(client) for _ in range(n_streamers)])
    return sent


# --- Setup / teardown ---

def wait_for_health(base, timeout=180):
    # Generous: `import main` eagerly builds the Gemini embedding client and runs
    # create_all + the reaper against remote Neon, so cold startup is ~50s.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/api/health", timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def ensure_user(base):
    # Retry: a remote free-tier instance can drop the first connection (cold
    # start) or briefly rate-limit signup/login. Don't let a blip kill the run.
    last = None
    for _ in range(5):
        try:
            httpx.post(f"{base}/api/auth/signup",
                       json={"username": LOAD_USER, "password": LOAD_PASS}, timeout=30)
            r = httpx.post(f"{base}/api/auth/login",
                           json={"username": LOAD_USER, "password": LOAD_PASS}, timeout=30)
            if r.status_code == 200:
                d = r.json()
                return d["token"], d["user_id"]
            last = f"{r.status_code} {r.text[:120]}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(2)
    sys.exit(f"Could not auth the load-test user after retries: {last}")


def create_chat(base, token):
    r = httpx.post(f"{base}/api/chats/new",
                   headers={"Authorization": f"Bearer {token}"}, timeout=30)
    r.raise_for_status()
    return r.json()["chat"]["id"]


def seed_messages(chat_id, user_id, n):
    """Insert N realistic message rows so GET /api/chats/{id} measures a real
    payload, not an empty list. A stored assistant turn carries ~800 chars of
    text plus a sources list, and the endpoint returns every message in the
    chat — so payload size is a real part of what that read costs. Removed by
    cleanup (which deletes the chat's messages)."""
    from db.database import SessionLocal
    from db.models import ChatMessage
    sources = [{"filename": "Test Blob File.pdf", "doc_type": "Lender Fee Sheet",
                "page_start": 1, "page_end": 1, "text": "y" * 300}]
    db = SessionLocal()
    try:
        for i in range(n):
            db.add(ChatMessage(chat_id=chat_id, user_id=user_id, role="user",
                               content=f"Seeded question {i}: what is the origination charge?"))
            db.add(ChatMessage(chat_id=chat_id, user_id=user_id, role="assistant",
                               content="z" * 800, sources=sources))
        db.commit()
    finally:
        db.close()


def cleanup():
    """Delete the load-test user and all its rows straight from the DB."""
    from sqlalchemy import text
    from db.database import engine
    with engine.begin() as c:
        uid = c.execute(text("SELECT id FROM drs_accounts WHERE username=:u"),
                        {"u": LOAD_USER}).scalar()
        if uid is None:
            print("nothing to clean up.")
            return
        chat_ids = [r[0] for r in c.execute(
            text("SELECT id FROM drs_chat_sessions WHERE user_id=:u"), {"u": uid})]
        if chat_ids:
            c.execute(text("DELETE FROM drs_chat_messages WHERE chat_id = ANY(:ids)"),
                      {"ids": chat_ids})
        c.execute(text("DELETE FROM drs_refresh_tokens WHERE user_id=:u"), {"u": uid})
        c.execute(text("DELETE FROM drs_chat_sessions WHERE user_id=:u"), {"u": uid})
        c.execute(text("DELETE FROM drs_login_failures WHERE username=:u"), {"u": LOAD_USER})
        c.execute(text("DELETE FROM drs_accounts WHERE id=:u"), {"u": uid})
    print(f"cleaned up load-test user {LOAD_USER} (id {uid}) and its rows.")


# --- Idle vs saturated report ---

def _row(label, idle, sat):
    if not idle or not sat:
        return f"  {label:8} {'no data':>50}"
    i50, i95, s50, s95 = pctl(idle, 50), pctl(idle, 95), pctl(sat, 50), pctl(sat, 95)
    ratio = s95 / i95 if i95 else float("nan")
    flag = "  <-- degraded" if ratio >= 2 else ""
    return (f"  {label:8} p50 {i50:6.0f} -> {s50:6.0f}ms   "
            f"p95 {i95:6.0f} -> {s95:6.0f}ms   x{ratio:.1f}{flag}")


def report(idle, sat, args, sent):
    print("\n" + "=" * 76)
    print(f"API UNDER LOAD  ({args.messages} streaming answers @ {args.msg_seconds}s each, "
          f"{args.concurrency} browse clients, mix={args.mix})")
    print("=" * 76)
    print(f"  {'':8} {'idle':>19}   {'saturated':>22}")
    for name in ("health", "chats", "chat", "login"):
        print(_row(name, idle["lat"][name], sat["lat"][name]))

    n_idle = sum(len(v) for v in idle["lat"].values())
    n_sat = sum(len(v) for v in sat["lat"].values())
    print(f"\n  Browse throughput  {n_idle / idle['elapsed']:.0f} -> {n_sat / sat['elapsed']:.0f} req/s")
    print(f"  Answers streamed   {sent} during the saturated phase")
    print(f"  Throttled (429)    {idle['throttled']} idle, {sat['throttled']} saturated")
    print(f"  Errors             {idle['errors']['count']} idle, {sat['errors']['count']} saturated")
    for s in (idle["errors"]["samples"] + sat["errors"]["samples"])[:5]:
        print(f"    {s}")

    out_dir = os.path.join(BASE_DIR, "load_test_results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"api_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(), "config": vars(args),
            "answers_streamed": sent,
            "idle": {k: {"p50": _num(pctl(v, 50)), "p95": _num(pctl(v, 95)), "n": len(v)}
                     for k, v in idle["lat"].items()},
            "saturated": {k: {"p50": _num(pctl(v, 50)), "p95": _num(pctl(v, 95)), "n": len(v)}
                          for k, v in sat["lat"].items()},
            "throttled": {"idle": idle["throttled"], "saturated": sat["throttled"]},
            "errors": {"idle": idle["errors"]["count"], "saturated": sat["errors"]["count"]},
        }, f, indent=2)
    print(f"\n  Report: {path}")
    print("  Simulated generation latency — compare the ratio, not absolute ms.")


# --- Ramp — step concurrency up until it breaks ---

def _agg(lat, keys):
    out = []
    for k in keys:
        out.extend(lat.get(k, []))
    return out


def ramp_mode(base, token, chat_id, levels, duration, stop_pct, mix):
    """Step browse concurrency up and watch where p95 climbs or errors appear —
    the read-path ceiling. Only browse is ramped; /message is rate-limited per
    IP, so message concurrency can't be measured honestly from one machine
    (use --calibrate for real per-message latency)."""
    print(f"RAMP against {base}  (mix={mix}: health + chats + chat"
          f"{' + login' if mix == 'all' else ''})\n")
    login_hdr = "  login p95" if mix == "all" else ""
    # Warm the pool so the first level (the baseline for the 3x-degraded flag)
    # isn't inflated by cold Neon connections.
    asyncio.run(_hammer(base, token, chat_id, levels[0], 4, mix))

    print(f"  {'clients':>7} | {'fast p50':>9} {'fast p95':>9} | {'req/s':>6} | {'err%':>5}{login_hdr}")
    print("  " + "-" * (58 + len(login_hdr)))

    rows, baseline_p95 = [], None
    for c in levels:
        res = asyncio.run(_hammer(base, token, chat_id, c, duration, mix))
        fast = _agg(res["lat"], ("health", "chats", "chat"))
        p50, p95 = pctl(fast, 50), pctl(fast, 95)
        lp95 = pctl(res["lat"]["login"], 95)
        n_ok = sum(len(v) for v in res["lat"].values())
        n_total = n_ok + res["errors"]["count"]
        err_pct = (res["errors"]["count"] / n_total * 100) if n_total else 0.0
        rps = n_total / res["elapsed"] if res["elapsed"] else 0.0
        if baseline_p95 is None and fast:
            baseline_p95 = p95
        degraded = bool(baseline_p95) and p95 > 3 * baseline_p95
        flag = "  <-- ERRORS" if err_pct > 1 else ("  <-- latency degrading" if degraded else "")
        login_col = f"  {lp95:8.0f}ms" if mix == "all" else ""
        print(f"  {c:>7} | {p50:>7.0f}ms {p95:>7.0f}ms | {rps:>6.0f} | {err_pct:>4.1f}%{login_col}{flag}")
        rows.append({"clients": c, "fast_p50_ms": _num(p50), "fast_p95_ms": _num(p95),
                     "login_p95_ms": _num(lp95), "req_s": round(rps),
                     "error_pct": round(err_pct, 1), "throttled": res["throttled"],
                     "err_samples": res["errors"]["samples"][:3]})
        if err_pct > stop_pct:
            print(f"  Error rate over {stop_pct:g}% at {c} clients — stopping the ramp.")
            break

    baseline = rows[0]["fast_p95_ms"] if rows else None
    healthy = [r for r in rows if r["error_pct"] < 1
               and baseline and r["fast_p95_ms"] and r["fast_p95_ms"] < 3 * baseline]
    ceiling = healthy[-1]["clients"] if healthy else 0
    print(f"\n  Estimated healthy ceiling: ~{ceiling} concurrent browse clients")
    print(f"  SLO: <1% errors AND fast-path p95 < 3x the {rows[0]['clients'] if rows else '?'}-client baseline")

    out_dir = os.path.join(BASE_DIR, "load_test_results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ramp_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"base": base, "duration": duration, "mix": mix,
                   "estimated_ceiling": ceiling, "levels": rows}, f, indent=2)
    print(f"  Report: {path}")


# --- Calibrate — a few REAL messages, to check the stub's timing model ---

def calibrate(n, base):
    """Ingest Test Blob File.pdf, then send N real messages through the running
    app (real rewrite + Pinecone + Gemini) and report latency. Costs money (a
    few Flash calls each) and a Textract ingest (~45s)."""
    if not os.path.exists(PDF):
        sys.exit(f"no PDF at {PDF} — calibrate needs a document to ingest.")
    token, _ = ensure_user(base)
    auth = {"Authorization": f"Bearer {token}"}
    chat_id = create_chat(base, token)

    print(f"Uploading {os.path.basename(PDF)} and waiting for ingest...")
    with open(PDF, "rb") as f:
        httpx.post(f"{base}/api/chats/{chat_id}/document",
                   files={"file": (os.path.basename(PDF), f, "application/pdf")},
                   headers=auth, timeout=300)
    t0, status = time.monotonic(), "processing"
    while status == "processing" and time.monotonic() - t0 < 900:
        time.sleep(8)
        status = httpx.get(f"{base}/api/chats/{chat_id}/status",
                           headers=auth, timeout=30).json()["status"]
    if status != "ready":
        cleanup()
        sys.exit(f"ingest did not finish (status={status})")
    print(f"  ready in {int(time.monotonic() - t0)}s")

    took = []
    print(f"Sending {n} real message(s) ...")
    for i in range(n):
        t = time.monotonic()
        with httpx.stream("POST", f"{base}/api/chats/{chat_id}/message",
                          json={"question": Q, "num_chunks": 6, "alpha": 0.5},
                          headers=auth, timeout=120) as r:
            for _ in r.iter_lines():
                pass
        dt = time.monotonic() - t
        took.append(dt)
        print(f"  message {i + 1}: {dt:.1f}s")
    print("\n" + "=" * 60)
    print(f"REAL MESSAGE LATENCY  (n={n})")
    print("=" * 60)
    print(f"  p50 {pctl(took, 50):.1f}s   p95 {pctl(took, 95):.1f}s   "
          f"min {min(took):.1f}s   max {max(took):.1f}s")
    print("\n  Feed a value near p50 to --msg-seconds for a realistic run.")
    cleanup()


# --- Main ---

def _spawn_server(port, msg_seconds):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # Unbuffered so the server's startup log survives a terminate() — otherwise
    # block buffering swallows a startup traceback and the log reads empty.
    env["PYTHONUNBUFFERED"] = "1"
    os.makedirs(os.path.join(BASE_DIR, "load_test_results"), exist_ok=True)
    log_path = os.path.join(BASE_DIR, "load_test_results", "server.log")
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--serve",
         "--port", str(port), "--msg-seconds", str(msg_seconds)],
        env=env, cwd=BASE_DIR, stdout=log, stderr=subprocess.STDOUT)
    return proc, log, log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--concurrency", type=int, default=20, help="browse clients")
    ap.add_argument("--messages", type=int, default=10, help="concurrent streaming answers")
    ap.add_argument("--duration", type=float, default=15, help="seconds per phase")
    ap.add_argument("--msg-seconds", type=float, default=2.0, help="simulated generation time")
    ap.add_argument("--mix", choices=("read", "all"), default="read",
                    help="'read' = realistic browse (no per-cycle login); 'all' adds a "
                         "rate-limited login every cycle to observe the limiter")
    ap.add_argument("--seed-messages", type=int, default=0, metavar="N",
                    help="insert N message rows so GET /api/chats/{id} measures a real payload")
    ap.add_argument("--calibrate", type=int, metavar="N", help="send N REAL messages (costs money)")
    ap.add_argument("--base", default=None, help="target a running server instead of spawning one")
    ap.add_argument("--ramp", action="store_true", help="ramp browse concurrency to find the knee")
    ap.add_argument("--levels", default="5,15,30,50,100", help="concurrency levels for --ramp")
    ap.add_argument("--ramp-stop-pct", type=float, default=25,
                    help="stop the ramp once a level's error rate exceeds this percent")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args()

    if args.serve:
        return serve_mode(args.port, args.msg_seconds)
    if args.cleanup:
        return cleanup()
    if args.calibrate:
        return calibrate(args.calibrate, args.base or "http://127.0.0.1:8000")

    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    # Ramp against a live server: no local spawn, no seeding a real deployment.
    if args.ramp and args.base:
        token, user_id = ensure_user(args.base)
        chat_id = create_chat(args.base, token)
        return ramp_mode(args.base, token, chat_id, levels, args.duration,
                         args.ramp_stop_pct, args.mix)

    base = f"http://127.0.0.1:{args.port}"
    proc, log, log_path = _spawn_server(args.port, args.msg_seconds)
    try:
        if not wait_for_health(base):
            sys.exit("app did not come up — see load_test_results/server.log")
        print(f"app up on {base} (retrieval + LLM stubbed)")
        token, user_id = ensure_user(base)
        chat_id = create_chat(base, token)
        if args.seed_messages:
            seed_messages(chat_id, user_id, args.seed_messages)
            print(f"seeded {args.seed_messages} message(s) into the browse chat")

        if args.ramp:
            return ramp_mode(base, token, chat_id, levels, args.duration,
                             args.ramp_stop_pct, args.mix)

        # Prime the connection pool first: the idle phase runs right after a ~50s
        # cold start, and the pool's first Neon connections (SSL handshakes) would
        # otherwise land in the idle baseline and make it look slower than the
        # saturated phase. Warm it, discard the numbers.
        print("Warming the connection pool...")
        asyncio.run(_hammer(base, token, chat_id, args.concurrency, 4, args.mix))

        print(f"Phase 1/2: idle, {args.concurrency} browse clients for {args.duration}s...")
        idle = asyncio.run(_hammer(base, token, chat_id, args.concurrency, args.duration, args.mix))

        print(f"Phase 2/2: saturated, {args.messages} answers streaming + browse load...")
        async def saturated():
            stop_evt = asyncio.Event()
            msg_task = asyncio.create_task(_message_load(base, token, chat_id, args.messages, stop_evt))
            await asyncio.sleep(2)   # let the message load ramp up
            sat = await _hammer(base, token, chat_id, args.concurrency, args.duration, args.mix)
            stop_evt.set()
            sent = await msg_task
            return sat, sent
        sat, sent = asyncio.run(saturated())

        report(idle, sat, args, sent)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        print(f"  Server log: {log_path}")
        cleanup()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        assert pctl([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
        assert pctl([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
        assert math.isnan(pctl([], 50))
        assert _num(float("nan")) is None and _num(None) is None and _num(3.14159) == 3.1
        print("selftest ok")
    else:
        main()
