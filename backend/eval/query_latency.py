"""Latency, TTFT and cost per query for the 120 loan packet questions, asked one
at a time through the app's chat path: Neon reads, cached retriever, rewrite,
hybrid retrieval with linked evidence and a streamed gemini-3.5-flash-lite
answer. No judging. Uses the index left by loan_packet_generate.py.

    python backend/eval/query_latency.py
"""
import json
import os
import sys
import time

# Same model as the evaluation, with no failover. Set before the router reads it.
os.environ["GEMINI_CHAT_MODEL"] = "gemini-3.5-flash-lite"
os.environ["ALLOW_FAILOVER"] = "0"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND, ".env"))

from core.answer_generator import build_sources, stream_answer
from core.document_store import EnhancedDocumentStoreHybrid
from core.query_rewriter import rewrite_standalone
from db.database import SessionLocal
from db.models import ChatMessage, ChatSession, now_ist
from llm.llm_router import embed_model, estimate_cost_usd, llm

K = 6
MODEL = os.environ["GEMINI_CHAT_MODEL"]
QUESTIONS = os.path.join(BACKEND, "eval", "loan_packet_questions.json")
INDEX_STATE = os.path.join(REPO, "results", "loan_packet_index.json")
OUT = os.path.join(REPO, "results", "query_latency_120.json")


def pctl(xs, p):
    """Nearest-rank percentile."""
    xs = sorted(xs)
    return xs[min(len(xs), max(1, -(-p * len(xs) // 100))) - 1]


def ask(store, chat_id, question):
    """One question as main.py's message endpoint answers it, minus the save."""
    started = time.monotonic()
    db = SessionLocal()
    try:
        db.query(ChatSession).filter(ChatSession.id == chat_id, ChatSession.user_id == 0).first()
        since = now_ist().replace(hour=0, minute=0, second=0, microsecond=0)
        db.query(ChatMessage).filter(ChatMessage.user_id == 0, ChatMessage.role == "user",
                                     ChatMessage.created_at >= since).count()
        history = [{"role": m.role, "content": m.content} for m in
                   db.query(ChatMessage).filter(ChatMessage.chat_id == chat_id)
                   .order_by(ChatMessage.id.desc()).limit(6).all()][::-1]
    finally:
        db.close()
    query = rewrite_standalone(question, history)
    retrieved = store.retrieve_only(query, k=K)
    build_sources(retrieved, K)
    first_token, parts = None, []
    for tok in stream_answer(query, retrieved):
        first_token = first_token or time.monotonic()
        parts.append(tok)
    usage = llm.last_stream_usage or {}
    if usage.get("model") != MODEL:
        raise RuntimeError(f"answered by {usage.get('model')!r}, expected {MODEL!r}")
    return {"latency_s": round(time.monotonic() - started, 2),
            "ttft_s": round(first_token - started, 2) if first_token else None,
            "chunks": len(retrieved), "usage": usage,
            "cost_usd": round(estimate_cost_usd(usage), 6), "answer": "".join(parts).strip()}


def main():
    with open(QUESTIONS, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    with open(INDEX_STATE, encoding="utf-8") as f:
        state = json.load(f)
    store = EnhancedDocumentStoreHybrid.rehydrate(
        namespace=state["namespace"], chat_id=state["chat_id"], bm25_params=state["bm25_params"],
        doc_stats=state["doc_stats"], embed_model=embed_model)

    rows = []
    for q in questions:
        try:
            row = {"id": q["id"], **ask(store, state["chat_id"], q["question"])}
            if not (row["answer"] and row["usage"]):
                row["error"] = "empty answer or no usage"
        except Exception as e:
            row = {"id": q["id"], "error": f"{type(e).__name__}: {e}"[:300]}
        rows.append(row)
        print(row["id"], row.get("latency_s"), row.get("cost_usd"), row.get("error", ""), flush=True)

    ok = [r for r in rows if "error" not in r]
    lat, cost = [r["latency_s"] for r in ok], [r["cost_usd"] for r in ok]
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    summary = {"cases": len(rows), "errors": len(rows) - len(ok),
               "latency_s": {**{f"p{p}": pctl(lat, p) for p in (50, 95, 99)},
                             "mean": round(sum(lat) / len(lat), 2)},
               "ttft_mean_s": round(sum(ttft) / len(ttft), 2),
               "cost_per_query_usd": {**{f"p{p}": pctl(cost, p) for p in (50, 95, 99)},
                                      "mean": round(sum(cost) / len(cost), 6)}}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "model": MODEL, "results": rows}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
