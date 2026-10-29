"""Generate answers for the 120-question loan packet eval through the app's own
retrieval and prompt. Ingests the fixture once, then answers in batches of 25,
saving after each batch so a rerun only retries unfinished questions.

    python backend/eval/loan_packet_generate.py
"""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# The eval answers with gemini-3.5-flash-lite only: no silent failover to another
# provider. Set before the router reads its environment.
os.environ["GEMINI_CHAT_MODEL"] = "gemini-3.5-flash-lite"
os.environ["ALLOW_FAILOVER"] = "0"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(BACKEND)
sys.path.insert(0, BACKEND)

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND, ".env"))

from core.answer_generator import build_prompt, build_sources
from core.document_store import EnhancedDocumentStoreHybrid
from core.evidence import source_block
from core.query_rewriter import rewrite_standalone
from llm.llm_router import embed_model, estimate_cost_usd, llm

MODEL = os.environ["GEMINI_CHAT_MODEL"]
QUESTIONS = os.path.join(BACKEND, "eval", "loan_packet_questions.json")
INDEX_STATE = os.path.join(REPO, "results", "loan_packet_index.json")
OUT = os.path.join(REPO, "results", "loan_packet_answers_v3.json")
NAMESPACE, CHAT_ID = "drs-eval", "loan-packet-mixed-v1"
K, ALPHA, BATCH = 6, 0.5, 25


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def ingest(pdf, fixture_sha):
    """Index the fixture once (Textract, as in the app) and keep what is needed
    to rehydrate it, plus how long each ingest stage took."""
    store = EnhancedDocumentStoreHybrid(namespace=NAMESPACE, chat_id=CHAT_ID, alpha=ALPHA)
    store.retriever.delete_chat()  # never mix vectors from an earlier ingest
    marks = [("start", time.perf_counter())]
    ok, stats = store.process_pdf(pdf, filename=os.path.basename(pdf), embed_model=embed_model,
                                  on_stage=lambda key: marks.append((key, time.perf_counter())))
    marks.append(("done", time.perf_counter()))
    if not ok:
        store.retriever.delete_chat()
        raise RuntimeError(f"Ingest failed: {stats.get('error')}")
    stage_s = {name: round(marks[i + 1][1] - t, 2) for i, (name, t) in enumerate(marks[:-1])
               if name != "start"}
    state = {"fixture_sha256": fixture_sha, "namespace": NAMESPACE, "chat_id": CHAT_ID,
             "extract_method": os.getenv("EXTRACT_METHOD", "textract"),
             "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "stage_seconds": stage_s, "total_seconds": round(marks[-1][1] - marks[0][1], 2),
             "doc_stats": stats, "bm25_params": store.export_bm25_params()}
    save(INDEX_STATE, state)
    print(f"Ingested {stats.get('total_chunks')} chunks in {state['total_seconds']}s: {stage_s}")
    return state


def answer(store, q):
    """One question through the app path: rewrite (a no-op without history),
    hybrid retrieval limited to six ranked matches, then the production prompt."""
    row = {key: q[key] for key in ("id", "question", "expected_answer",
                                   "pages", "answerable")}
    try:
        search_query = rewrite_standalone(q["question"], [])
        retrieved = [(c, score) for c, score in store.retrieve_only(search_query, k=K)
                     if score > 0][:K]
        if len(retrieved) != K:
            raise RuntimeError(f"expected {K} ranked chunks, got {len(retrieved)}")
        row.update(
            search_query=search_query,
            sources=[{key: s[key] for key in ("doc_type", "pages", "relevance")}
                     for s in build_sources(retrieved, K)],
            retrieved_chunks=[{"chunk_id": c.chunk_id, "doc_type": c.doc_type,
                               "pages": [c.page_start + 1, c.page_end + 1],
                               "score": score, "linked": not score} for c, score in retrieved],
            retrieved_contexts=[source_block(c) for c, _ in retrieved])
        reply = llm.complete(build_prompt(search_query, retrieved))
        usage = reply.usage or {}
        if not reply.text:
            raise RuntimeError("model returned no text")
        if usage.get("model") != MODEL:
            raise RuntimeError(f"answered by {usage.get('model')!r}, expected {MODEL!r}")
        row.update(response=reply.text, usage=usage, cost_usd=estimate_cost_usd(usage),
                   gen_status="complete", error=None)
    except Exception as e:
        row.update(response=None, gen_status="error", error=f"{type(e).__name__}: {e}"[:500])
    return row


def main():
    with open(QUESTIONS, encoding="utf-8") as f:
        dataset = json.load(f)
    questions = dataset["questions"]
    assert len(questions) == 120 and len({q["id"] for q in questions}) == 120
    pdf = os.path.join(REPO, dataset["fixture"])
    fixture_sha = sha256(pdf)
    assert fixture_sha == dataset["fixture_sha256"], "fixture PDF changed since the questions were written"

    state = None
    if os.path.exists(INDEX_STATE):
        with open(INDEX_STATE, encoding="utf-8") as f:
            state = json.load(f)
        if state.get("fixture_sha256") != fixture_sha:
            state = None
    if state is None:
        state = ingest(pdf, fixture_sha)
    # Rehydrate even after a fresh ingest: the app answers from a rehydrated store.
    store = EnhancedDocumentStoreHybrid.rehydrate(
        namespace=NAMESPACE, chat_id=CHAT_ID, bm25_params=state["bm25_params"],
        doc_stats=state["doc_stats"], embed_model=embed_model, alpha=ALPHA)

    rows = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            previous = json.load(f)
        if (previous["meta"].get("fixture_sha256") == fixture_sha
                and previous["meta"].get("generator_model") == MODEL
                and previous["meta"].get("evidence_mode") == "top6_ranked_only"
                and previous["meta"].get("dataset_sha256") == sha256(QUESTIONS)):
            rows = {r["id"]: r for r in previous["results"]}
    todo = [q for q in questions if rows.get(q["id"], {}).get("gen_status") != "complete"]
    print(f"{len(questions) - len(todo)} already complete; answering {len(todo)} "
          f"in batches of {BATCH} with {MODEL}")

    meta = {"fixture": dataset["fixture"], "fixture_sha256": fixture_sha,
            "generator_model": MODEL, "k": K, "alpha": ALPHA, "batch_size": BATCH,
            "namespace": NAMESPACE, "chat_id": CHAT_ID,
            "evidence_mode": "top6_ranked_only", "dataset_sha256": sha256(QUESTIONS)}
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            for row in pool.map(lambda q: answer(store, q), batch):
                rows[row["id"]] = row
        ordered = [rows[q["id"]] for q in questions if q["id"] in rows]
        save(OUT, {"meta": meta, "results": ordered})
        errors = sum(rows[q["id"]]["gen_status"] != "complete" for q in batch)
        print(f"Batch {start // BATCH + 1}: {len(batch) - errors}/{len(batch)} complete")

    results = [rows[q["id"]] for q in questions]
    done = [r for r in results if r["gen_status"] == "complete"]
    costs = [r["cost_usd"] for r in done]
    priced = [c for c in costs if c is not None]
    print(f"\n{len(done)}/120 complete, {120 - len(done)} errors"
          + (" (rerun to retry them)" if len(done) < 120 else ""))
    print(f"Generation cost: ${sum(priced):.4f} over {len(priced)} priced answers"
          + (f"; {len(costs) - len(priced)} could not be priced" if len(priced) < len(costs) else ""))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
