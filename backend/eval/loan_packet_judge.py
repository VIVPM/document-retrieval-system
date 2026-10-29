"""Score results/loan_packet_answers_v3.json with Ragas (answer correctness, context
precision, context recall, faithfulness), judged by gemini-3.5-flash-lite, in
batches of 25 questions. Saves after each batch; a rerun only rescores questions
that are not complete. Pass question ids to score just those.

Runs in its own environment so Ragas' pins stay out of the app's:
    backend/eval/.venv/Scripts/python.exe backend/eval/loan_packet_judge.py
"""
import asyncio
import contextvars
import csv
import json
import math
import os
import random
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
import instructor
from ragas.embeddings import GoogleEmbeddings
from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import AnswerCorrectness, ContextPrecision, ContextRecall, Faithfulness

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(BACKEND)
load_dotenv(os.path.join(BACKEND, ".env"))

JUDGE = "gemini-3.5-flash-lite"
EMBEDDINGS = "gemini-embedding-001"
PRICE = {"in": 0.30, "out": 2.50}  # USD per 1M tokens, as llm_router.PRICE_PER_MTOK; output includes thinking
ANSWERS = os.path.join(REPO, "results", "loan_packet_answers_v3.json")
OUT = os.path.join(REPO, "results", "loan_packet_ragas_top6_v3_judge35lite.csv")
BATCH = 25
MAX_JUDGE_CALLS = 16  # concurrent judge requests across a batch
METRICS = ("answer_correctness", "context_precision", "context_recall", "faithfulness")
FIELDS = ["question_id", "answerable", "question", "expected_answer", "answer",
          *METRICS, "judge_calls", "judge_input_tokens", "judge_output_tokens", "judge_cost_usd",
          "status", "error"]

_calls: contextvars.ContextVar[list | None] = contextvars.ContextVar("judge_calls", default=None)


def judge_client():
    """A Gemini client whose async calls are rate-capped, retried on transient
    errors and counted per question, so judge cost is measured, not guessed."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    models = client.aio.models
    original, gate = models.generate_content, asyncio.Semaphore(MAX_JUDGE_CALLS)

    async def generate_content(*args, **kwargs):
        async with gate:
            for attempt in range(5):
                try:
                    response = await original(*args, **kwargs)
                    break
                except genai_errors.APIError as e:
                    if e.code not in (429, 500, 503, 504) or attempt == 4:
                        raise
                    await asyncio.sleep(min(30, 2 ** attempt) * random.uniform(0.5, 1.0))
        m, log = response.usage_metadata, _calls.get()
        if log is not None and m is not None:
            log.append(((m.prompt_token_count or 0),
                        (m.candidates_token_count or 0) + (m.thoughts_token_count or 0)))
        return response

    models.generate_content = generate_content
    return client


def build_metrics(client):
    llm = InstructorLLM(client=instructor.from_genai(client, use_async=True), model=JUDGE,
                        provider="google", temperature=0.0, max_tokens=4096)
    return {"answer_correctness": AnswerCorrectness(
                llm=llm, embeddings=GoogleEmbeddings(client=client, model=EMBEDDINGS)),
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
            "faithfulness": Faithfulness(llm=llm)}


def top_six_contexts(row):
    """Score only the first six ranked search matches, never linked evidence."""
    chunks, contexts = row["retrieved_chunks"], row["retrieved_contexts"]
    assert len(chunks) == len(contexts), "chunk/context alignment mismatch"
    selected = [text for chunk, text in zip(chunks, contexts)
                if not chunk["linked"] and chunk["score"] > 0][:6]
    assert len(selected) == 6, f"{row['id']}: expected six ranked contexts"
    return selected


async def score(row, metrics):
    """Judge answers against the selected six ranked search matches."""
    log = []
    _calls.set(log)
    q, a, ref = row["question"], row["response"], row["expected_answer"]
    ctx = top_six_contexts(row)
    results = await asyncio.gather(
        metrics["answer_correctness"].ascore(user_input=q, response=a, reference=ref),
        metrics["context_precision"].ascore(user_input=q, reference=ref, retrieved_contexts=ctx),
        metrics["context_recall"].ascore(user_input=q, retrieved_contexts=ctx, reference=ref),
        metrics["faithfulness"].ascore(user_input=q, response=a, retrieved_contexts=ctx),
        return_exceptions=True)
    out = {"question_id": row["id"], "answerable": row["answerable"],
           "question": q, "expected_answer": ref, "answer": a}
    errors = []
    for name, result in zip(METRICS, results):
        if isinstance(result, BaseException):
            out[name] = ""
            errors.append(f"{name}: {type(result).__name__}: {result}"[:300])
        else:
            value = result.value
            # NaN means Ragas found nothing to score (e.g. no claims in a refusal), not a failed call.
            out[name] = "" if value is None or (isinstance(value, float) and math.isnan(value)) else value
    tokens_in, tokens_out = sum(t[0] for t in log), sum(t[1] for t in log)
    out.update(judge_calls=len(log), judge_input_tokens=tokens_in, judge_output_tokens=tokens_out,
               judge_cost_usd=(round((tokens_in * PRICE["in"] + tokens_out * PRICE["out"]) / 1e6, 6)
                               if PRICE else ""),
               status="error" if errors else "complete", error=" | ".join(errors))
    return out


def save(rows, order):
    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[i] for i in order if i in rows)
    os.replace(tmp, OUT)


async def main(only):
    with open(ANSWERS, encoding="utf-8") as f:
        answers = json.load(f)
    rows_in = answers["results"]
    assert len(rows_in) == 120 and all(r["gen_status"] == "complete" for r in rows_in), \
        "generate all 120 answers before judging"
    assert answers["meta"].get("evidence_mode") == "top6_ranked_only", \
        "generation and judging must use the same six chunks"
    for row in rows_in:
        assert len(row["retrieved_contexts"]) == 6
        top_six_contexts(row)  # Validate all contexts before any paid judge calls.
    order = [r["id"] for r in rows_in]

    done = {}
    if os.path.exists(OUT):
        with open(OUT, newline="", encoding="utf-8") as f:
            done = {r["question_id"]: r for r in csv.DictReader(f)}
    if only:
        todo = [r for r in rows_in if r["id"] in only]
    else:
        todo = [r for r in rows_in if done.get(r["id"], {}).get("status") != "complete"]
    print(f"Judging {len(todo)} question(s) with {JUDGE} in batches of {BATCH}, top 6 ranked contexts only "
          f"(generator: {answers['meta']['generator_model']})")

    metrics = build_metrics(judge_client())
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        for out in await asyncio.gather(*(score(r, metrics) for r in batch)):
            done[out["question_id"]] = out
        save(done, order)
        failed = [r["id"] for r in batch if done[r["id"]]["status"] != "complete"]
        print(f"Batch {start // BATCH + 1}: {len(batch) - len(failed)}/{len(batch)} complete"
              + (f"; errors: {failed}" if failed else ""))

    rows = [done[i] for i in order if i in done]
    complete = [r for r in rows if r["status"] == "complete"]
    print(f"\n{len(complete)}/120 complete" + ("" if len(complete) == 120 else " (rerun to retry the rest)"))
    for name in METRICS:
        values = [float(r[name]) for r in complete if r[name] != ""]
        if values:
            print(f"{name:20s} {sum(values) / len(values):.3f}  (n={len(values)})")
    print(f"Judge cost: ${sum(float(r['judge_cost_usd']) for r in rows):.4f} over "
          f"{sum(int(r['judge_calls']) for r in rows)} calls; embedding calls not priced")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    asyncio.run(main(set(sys.argv[1:])))
