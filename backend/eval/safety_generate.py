"""Generate the safety responses with exactly six ranked chunks.
Run with backend/.venv/Scripts/python.exe. Uses the existing packet index.
"""
import json
import os
import time

import loan_packet_generate as gen

OUT = os.path.join(gen.REPO, "results", "current", "safety_answers.json")


def main():
    with open(os.path.join(gen.BACKEND, "eval", "safety_questions.json"), encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    with open(gen.INDEX_STATE, encoding="utf-8") as f:
        state = json.load(f)
    with open(gen.QUESTIONS, encoding="utf-8") as f:
        fixture = json.load(f)
    assert state["fixture_sha256"] == fixture["fixture_sha256"]
    assert len(questions) == 20
    store = gen.EnhancedDocumentStoreHybrid.rehydrate(
        namespace=state["namespace"], chat_id=state["chat_id"], bm25_params=state["bm25_params"],
        doc_stats=state["doc_stats"], embed_model=gen.embed_model, alpha=gen.ALPHA)
    rows = []
    for question in questions:
        q = {"id": question["id"], "question": question["question"].replace(
            "{other_chat_id}", "safety-other-account-chat-not-provisioned"),
            "expected_answer": question["expected_answer"], "answerable": question["id"] == "PII09", "pages": []}
        started = time.monotonic()
        row = gen.answer(store, q)
        row.update(latency_s=round(time.monotonic() - started, 3), category=question["category"],
                   fixture_setup_exercised=False, expected_behavior=question["expected_behavior"])
        rows.append(row)
        gen.save(OUT, {"meta": {"generator_model": gen.MODEL, "evidence_mode": "top6_ranked_only",
                               "fixture_sha256": state["fixture_sha256"], "k": gen.K,
                               "notes": "Prompt-response checks only. Cross-account fixtures and injected document not provisioned. Buffered generation; no TTFT measurement."},
                       "results": rows})
        print(row["id"], row["gen_status"], row["latency_s"], row.get("error"), flush=True)
    assert all(r["gen_status"] == "complete" and len(r["retrieved_contexts"]) == 6 for r in rows)


if __name__ == "__main__":
    main()
