"""
Compares two answer models on a fixed question set, graded by gemini-2.5-pro.

Retrieval is computed once and shared, so both arms see identical context.
Prefer model_sweep.py for model selection — this set is small and has been
iterated on. Kept because it exercises citations and prose, which string
matching cannot.
"""
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter

# Broad on purpose: a miss here would look like a model failure.
REFUSAL = re.compile(
    r"does not contain|doesn't contain|not contain|no information|not (?:be )?found|"
    r"cannot determine|can't determine|not (?:explicitly )?(?:present|available|stated|provided)|"
    r"unable to (?:answer|determine)|not enough information|insufficient",
    re.I)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(BACKEND)
PDF = os.getenv("EVAL_PDF", os.path.join(REPO, "Test Blob File.pdf"))
sys.path.insert(0, BACKEND)

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND, ".env"))

from core.answer_generator import generate_answer_with_sources
from core.chunker import chunk_by_structure
from core.pdf_processor import extract_and_analyze_pdf
from llm.llm_router import embed_model, llm

# (label, model, thinking_budget).
ARM_SPECS = [
    ("flash+think2048", "gemini-2.5-flash",      2048),
    ("flash+think0",    "gemini-2.5-flash",         0),
    ("flash-lite",      "gemini-2.5-flash-lite", 2048),
]
ARMS = [a[0] for a in ARM_SPECS]
SPEC = {a[0]: a for a in ARM_SPECS}
JUDGE = "gemini-2.5-pro"
K = 6
SEED = 20260730

# USD per 1M tokens. Thinking bills at the output rate.
PRICE = {
    "gemini-2.5-flash":      {"in": 0.30, "out": 2.50},
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
}

# `expect` is checked independently of the judge:
#   [..]  must state one of these strings
#   []    must refuse, the value is absent from the packet
#   None  judge only
# Every string must be grepped from the real extraction, never copied from a
# prompt's few-shot examples.
QUESTIONS = [
    ("What interest rate is being offered?",                     ["4.250", "4.25"]),
    ("What is the Underwriting Fee?",                            ["550"]),
    ("What is the Appraisal Fee?",                               ["525"]),
    ("Who pays the Lender's Title Insurance and how much is it?", ["650"]),
    ("What is the Total Estimated Funds needed to close?",        ["95,641.53"]),
    ("What is the Loan Amount?",                                  ["380,000"]),
    ("What is the monthly Principal & Interest payment?",         ["1,869.37"]),
    ("What is the Net Pay on the pay slip?",                      ["8000"]),
    ("What are the Total Earnings on the pay slip?",              ["8800"]),
    # Absent from the packet — a correct answer declines, a wrong one invents.
    ("When does the interest rate lock expire?",                   []),
    ("What is the Annual Percentage Rate (APR)?",                  []),
    ("What is the borrower's social security number?",             []),
    # Judge-only: no single correct string.
    ("What are the origination charges?",                          None),
    ("What is the sum of the Underwriting Fee and the Appraisal Fee?", None),
    ("List the closing costs and say which is the largest.",       None),
]

JUDGE_PROMPT = """You are grading several candidate answers to a question about \
a financial document. You are given the ONLY context that was available to any \
of them. They are in randomised order and you are not told which model wrote \
which — judge the text alone.

Grade strictly, but grade the right thing.

An answer that asserts a figure as if it were printed in the document, when it \
is not, is WRONG no matter how confident or well-written it is.

An answer that correctly says the context does not contain the information is \
GOOD, not a failure.

A figure the answer DERIVES from cited numbers, and which it labels as derived \
("calculated as", "the sum of", showing the arithmetic), is NOT a grounding \
failure. Judge it on two things only: is the arithmetic right, and are the \
inputs present in the context? If both hold, grounding is 5 even though the \
result is absent from the document. Reserve low grounding for figures that are \
invented, mis-scoped, or presented as quoted document values.

Listing the component values and stating their total are BOTH acceptable \
answers to a question about a group of charges. Do not mark one down for \
including the other.

CONTEXT
{context}

QUESTION
{question}

{answers}
Score EVERY answer independently on:
  correctness  0-5  does it answer the question with the right value(s) from the context
  grounding    0-5  is every figure and claim traceable to the context (5 = fully, 0 = invented)
  citation     0-2  does it name the document type and page it came from

Identical answers must receive identical scores.

Reply with ONLY this JSON, no prose and no code fences — one object per slot
letter shown above:
{{"scores": {{"A": {{"correctness":0,"grounding":0,"citation":0}}, ...}},
  "why": "one sentence on any answer that lost points"}}"""


def cosine(u, v):
    dot = sum(x * y for x, y in zip(u, v))
    return dot / (math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(y * y for y in v)) + 1e-12)


def cost(usage):
    p = PRICE.get(usage.get("model"))
    if not p:
        return 0.0
    billed_out = usage.get("output_tokens", 0) + usage.get("thinking_tokens", 0)
    return (usage.get("prompt_tokens", 0) * p["in"] + billed_out * p["out"]) / 1e6


# ── 1. Extract + chunk (real pipeline; also exercises flash-lite classify) ────
print(f"[extract] {os.path.basename(PDF)} via Docling on Modal")
t0 = time.time()
pages, docs = extract_and_analyze_pdf(PDF, filename=os.path.basename(PDF))
print(f"   {len(pages)} pages, {len(docs)} logical docs in {time.time()-t0:.0f}s")
for d in docs:
    print(f"     - {d.doc_type}  pages {d.page_start}-{d.page_end}")

chunks = [c for d in docs for c in chunk_by_structure(d)]
print(f"[chunk] {len(chunks)} chunks")
if not chunks:
    sys.exit("no chunks — extraction failed")

# ── 2. Embed once, select top-K per question ─────────────────────────────────
print(f"[embed] {len(chunks)} chunks + {len(QUESTIONS)} queries")
t0 = time.time()
cvecs = embed_model.encode([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")
qvecs = embed_model.encode([q for q, _ in QUESTIONS], task_type="RETRIEVAL_QUERY")
print(f"   {time.time()-t0:.0f}s")

# ── 3. Both arms, identical context; then judge ──────────────────────────────
rng = random.Random(SEED)
rows, wins, totals = [], Counter(), {m: Counter() for m in ARMS}
spend = {m: 0.0 for m in ARMS}

for i, (question, expect) in enumerate(QUESTIONS):
    ranked = sorted(zip(chunks, (cosine(qvecs[i], cv) for cv in cvecs)),
                    key=lambda t: t[1], reverse=True)[:K]
    context = "\n".join(
        f"[Source: {c.filename} | {c.doc_type} | Pages: {c.page_start}-{c.page_end}]\n{c.text}\n"
        for c, _ in ranked)

    # If retrieval did not surface the value, refusing is the correct answer.
    in_context = None if expect is None else any(e in context for e in expect)

    out = {}
    for arm in ARMS:
        _, model, think = SPEC[arm]
        t0 = time.time()
        r = generate_answer_with_sources(question, ranked, model=model,
                                         thinking_budget=think)
        u, ans = r.get("usage") or {}, r["answer"]
        if expect is None:
            hit = None
        elif in_context:
            hit = any(e in ans for e in expect)          # must state the value
        else:
            hit = bool(REFUSAL.search(ans))              # must decline to guess
        out[arm] = {"answer": ans, "secs": time.time() - t0, "usage": u,
                    "hit": hit}
        spend[arm] += cost(u)

    # Blind the judge: shuffle arms into slot letters afresh each question.
    shuffled = ARMS[:]
    rng.shuffle(shuffled)
    slots = {chr(ord("A") + n): arm for n, arm in enumerate(shuffled)}
    answers_block = "\n".join(f"ANSWER {ltr}\n{out[arm]['answer']}\n"
                              for ltr, arm in slots.items())
    verdict = {}
    try:
        raw = llm.complete(
            JUDGE_PROMPT.format(context=context, question=question,
                                answers=answers_block),
            # gemini-2.5-pro rejects thinking_budget=0 and draws thinking from
            # max_tokens, so both are generous here.
            model=JUDGE, temperature=0.0, max_tokens=16384, thinking_budget=-1,
        ).text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        verdict = json.loads(raw)
    except Exception as e:
        print(f"   ⚠️ judge failed on Q{i+1}: {type(e).__name__}: {e}")

    scores = verdict.get("scores") or {}
    for ltr, arm in slots.items():
        s = scores.get(ltr) or {}
        for metric in ("correctness", "grounding", "citation"):
            totals[arm][metric] += s.get(metric, 0)
    if not scores:
        wins["judge failed"] += 1

    rows.append((question, expect, out, slots, verdict, in_context))
    want = ("" if in_context is None else
            "  [value IS in context — must state it]" if in_context else
            "  [value NOT in context — must refuse]")
    print(f"\nQ{i+1}. {question}{want}")
    for arm in ARMS:
        o = out[arm]
        tag = "" if o["hit"] is None else ("  ✓expect" if o["hit"] else "  ✗EXPECT")
        sc = scores.get(next(l for l, a in slots.items() if a == arm)) or {}
        print(f"   {arm:16} {o['secs']:5.1f}s  "
              f"think={o['usage'].get('thinking_tokens',0):5}  "
              f"${cost(o['usage']):.5f}  "
              f"judge={sc.get('correctness','?')}/{sc.get('grounding','?')}"
              f"/{sc.get('citation','?')}{tag}")
        print(f"      {o['answer'][:140].replace(chr(10),' ')}")
    if verdict.get("why"):
        print(f"   why: {verdict['why'][:130]}")

# ── 4. Report ────────────────────────────────────────────────────────────────
n = len(QUESTIONS)
graded = [r for r in rows if r[4]]
hard = [(q, e, o) for q, e, o, _, _, _ in rows if e is not None]
must_state = sum(1 for r in rows if r[5] is True)
must_refuse = sum(1 for r in rows if r[5] is False)

print(f"\n{'='*74}\nRESULT over {n} questions — {len(graded)} judged, {len(hard)} with "
      f"ground truth ({must_state} must-state, {must_refuse} must-refuse)\n")
print(f"{'':16} {'correct/5':>10} {'ground/5':>9} {'cite/2':>7} "
      f"{'truth':>7} {'think':>7} {'secs':>6} {'$/q':>9}")
for arm in ARMS:
    t = totals[arm]
    hits = sum(1 for _, _, o in hard if o[arm]["hit"])
    think = sum(r[2][arm]["usage"].get("thinking_tokens", 0) for r in rows) / n
    secs = sum(r[2][arm]["secs"] for r in rows) / n
    print(f"{arm:16} {t['correctness']/max(len(graded),1):10.2f} "
          f"{t['grounding']/max(len(graded),1):9.2f} "
          f"{t['citation']/max(len(graded),1):7.2f} "
          f"{f'{hits}/{len(hard)}':>7} {think:7.0f} {secs:6.1f} "
          f"{spend[arm]/n:9.5f}")

if wins.get("judge failed"):
    print(f"\n⚠️ judge produced no scores on {wins['judge failed']} question(s)")

# Named per question: at temperature 0.3 the failing question moves between
# runs, which the aggregate hides.
print("\nground-truth misses:")
any_miss = False
for q, e, o, _, _, ic in rows:
    for arm in ARMS:
        if o[arm]["hit"] is False:
            any_miss = True
            print(f"  {arm:16} {'refused' if ic else 'guessed'} — {q[:52]}")
if not any_miss:
    print("  none")

base = spend[ARMS[-1]]
if base:
    print("\ncost vs " + ARMS[-1] + ": " + ", ".join(
        f"{a}={spend[a]/base:.1f}x" for a in ARMS))

# Next to the script, not the CWD — run from the repo root it otherwise drops a
# 32KB transcript into the project root.
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "model_eval_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump([{"question": q, "expect": e, "in_context": ic,
                "arms": o, "slots": s, "verdict": v}
               for q, e, o, s, v, ic in rows], f, indent=2, default=str)
print(f"\nfull transcript → {out_path}")
