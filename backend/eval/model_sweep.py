"""
Compares answer models on questions harvested from the document itself.

Every table row shaped `Label | ... | Number` becomes "What is the {Label}?"
with that number as ground truth, graded by string match. No LLM judge, and no
hand-picked questions, so neither can be tuned toward a result.

Set EVAL_MODELS, EVAL_TRIALS, EVAL_PDF to change what it runs on.
"""
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict

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
from llm.llm_router import embed_model

MODELS = (os.getenv("EVAL_MODELS") or
          "gemini-2.5-flash,gemini-2.5-flash-lite,"
          "gemini-3.6-flash,gemini-3.5-flash-lite").split(",")
TRIALS = int(os.getenv("EVAL_TRIALS", "2"))
K = 6

# USD per 1M tokens, from ai.google.dev/gemini-api/docs/pricing. Unlisted
# models print "?" rather than a guess. Note 3.5-flash-lite is priced at
# 2.5-FLASH rates.
PRICE = {
    "gemini-2.5-flash":      {"in": 0.30, "out": 2.50},
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
    "gemini-3.6-flash":      {"in": 1.50, "out": 7.50},
    "gemini-3.5-flash-lite": {"in": 0.30, "out": 2.50},
}

# Broad on purpose: a miss here reads as a model failing to refuse. Add
# phrasings rather than tightening it.
REFUSAL = re.compile(
    r"does not contain|doesn't contain|not contain|no information|not (?:be )?found|"
    r"cannot determine|can't determine|"
    r"not (?:explicitly |specifically )?(?:present|available|stated|provided|mentioned|"
    r"listed|included|specified|given|shown|disclosed)|"
    r"unable to (?:answer|determine|find)|not enough information|insufficient|"
    r"(?:does not|doesn't|do not|don't) (?:include|mention|list|specify|provide|state|show)|"
    r"no (?:mention|reference|record) of|is not in the (?:context|document)",
    re.I)

MONEY = re.compile(r"^\$?\s*([\d,]+(?:\.\d{1,2})?)\s*%?$")
# Must start with a letter, or cells like "$ 39.58 x 12 mth(s)" qualify.
GOOD_LABEL = re.compile(r"^[A-Za-z]")
# Form arithmetic markers: "Purchase Price (+)" is the field "Purchase Price".
TRIM = re.compile(r"\s*[:(]\s*[+\-]?\s*\)?\s*$|\s*\(\s*[+\-]\s*\)\s*$")

# A cell repeated across more rows than this is a column value (a payee, say),
# not a field name. Keeps the rule document-agnostic.
MAX_LABEL_REPEATS = 2


def cosine(u, v):
    dot = sum(x * y for x, y in zip(u, v))
    return dot / (math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(y * y for y in v)) + 1e-12)


def harvest(blocks):
    """Pull (label, value) pairs out of every table row."""
    pairs = []
    for kind, content in blocks:
        if kind != "table":
            continue
        # Row 0 is the header row (`Fee | Paid To | Paid By | Amount`); its
        # cells name columns, not fields.
        rows = [[c.strip() for c in r.split("|")] for r in content.split("\n")][1:]

        seen = Counter(c for row in rows for c in row
                       if c and not MONEY.match(c))

        def usable(cell):
            return (cell and GOOD_LABEL.match(cell) and not MONEY.match(cell)
                    and seen[cell] <= MAX_LABEL_REPEATS)

        for cells in rows:
            nums = [(i, MONEY.match(c)) for i, c in enumerate(cells)
                    if c and MONEY.match(c)]
            if not nums or len(cells) < 2:
                continue
            # Nearest preceding usable cell, which handles both a fee row and
            # a multi-column form row once repeated cells are excluded.
            for idx, m in nums:
                for j in range(idx - 1, -1, -1):
                    if usable(cells[j]):
                        pairs.append((TRIM.sub("", cells[j]), m.group(1)))
                        break
    return pairs


def build_questions(blocks):
    pairs = harvest(blocks)
    by_label = defaultdict(set)
    for label, value in pairs:
        by_label[label].add(value)

    qs, dropped = [], []
    for label in sorted(by_label):
        vals = by_label[label]
        if len(vals) > 1:                      # same label, different numbers
            dropped.append((label, sorted(vals)))
            continue
        value = next(iter(vals))
        if len(label) < 3 or len(label) > 60:
            continue
        # Accept the value with or without thousands separators / decimals.
        variants = {value, value.replace(",", "")}
        if value.endswith(".00"):
            variants.add(value[:-3])
        qs.append({"q": f"What is the {label}?", "expect": sorted(variants),
                   "kind": "lookup", "label": label})
    return qs, dropped


# Hand-written, because absent values cannot be harvested. Verified by grep.
ABSENT = [
    "What is the Annual Percentage Rate (APR)?",
    "What is the borrower's social security number?",
    "When does the interest rate lock expire?",
]

print(f"[extract] {os.path.basename(PDF)}")
pages, docs = extract_and_analyze_pdf(PDF, filename=os.path.basename(PDF))
chunks = [c for d in docs for c in chunk_by_structure(d)]
blocks = [(b.kind, b.content) for p in pages for b in p.blocks]
print(f"[chunk] {len(chunks)} chunks from {len(blocks)} blocks")

QUESTIONS, dropped = build_questions(blocks)
for a in ABSENT:
    QUESTIONS.append({"q": a, "expect": [], "kind": "absent", "label": "—"})

print(f"[generate] {len(QUESTIONS)} questions "
      f"({sum(1 for q in QUESTIONS if q['kind']=='lookup')} harvested, "
      f"{len(ABSENT)} hand-written absent), {len(dropped)} labels dropped as ambiguous")
for label, vals in dropped[:8]:
    print(f"    dropped: {label!r} → {vals}")
if len(QUESTIONS) < 10:
    sys.exit("too few questions harvested — check the extraction shape")

# ── Retrieval, computed ONCE and shared by every model ───────────────────────
cvecs = embed_model.encode([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")
qvecs = embed_model.encode([q["q"] for q in QUESTIONS], task_type="RETRIEVAL_QUERY")

for i, q in enumerate(QUESTIONS):
    ranked = sorted(zip(chunks, (cosine(qvecs[i], cv) for cv in cvecs)),
                    key=lambda t: t[1], reverse=True)[:K]
    q["ranked"] = ranked
    ctx = "\n".join(c.text for c, _ in ranked)
    # A value that was not retrieved is a retrieval failure, not a model one.
    q["retrieved"] = (q["kind"] == "absent") or any(e in ctx for e in q["expect"])

n_ret = sum(1 for q in QUESTIONS if q["kind"] == "lookup" and q["retrieved"])
n_look = sum(1 for q in QUESTIONS if q["kind"] == "lookup")
print(f"[retrieval] value present in top-{K} for {n_ret}/{n_look} lookups "
      f"— only these are scored against the models\n")

# ── Sweep ────────────────────────────────────────────────────────────────────
res = {m: Counter() for m in MODELS}
spend = {m: 0.0 for m in MODELS}
secs = {m: 0.0 for m in MODELS}
calls = {m: 0 for m in MODELS}
misses = defaultdict(list)

for model in MODELS:
    t_start = time.time()
    for q in QUESTIONS:
        if q["kind"] == "lookup" and not q["retrieved"]:
            continue
        for _ in range(TRIALS):
            t0 = time.time()
            r = generate_answer_with_sources(q["q"], q["ranked"], model=model)
            ans, u = r["answer"], (r.get("usage") or {})
            secs[model] += time.time() - t0
            calls[model] += 1
            p = PRICE.get(model)
            if p:
                spend[model] += (u.get("prompt_tokens", 0) * p["in"] +
                                 (u.get("output_tokens", 0) + u.get("thinking_tokens", 0))
                                 * p["out"]) / 1e6
            res[model]["think"] += u.get("thinking_tokens", 0)

            if q["kind"] == "absent":
                ok = bool(REFUSAL.search(ans))
                res[model]["absent_n"] += 1
                res[model]["absent_ok"] += ok
            else:
                ok = any(e in ans for e in q["expect"])
                res[model]["look_n"] += 1
                res[model]["look_ok"] += ok
            if not ok:
                misses[model].append((q["q"], q["expect"], ans[:88]))
    print(f"  {model:24} done in {time.time()-t_start:5.0f}s")

# ── Report ───────────────────────────────────────────────────────────────────
print(f"\n{'='*94}\nMODEL SWEEP — {len(QUESTIONS)} questions, {TRIALS} trials, "
      f"mechanical grading, no LLM judge\n")
print(f"{'model':24} {'lookup':>12} {'refuse':>10} {'overall':>10} "
      f"{'think/q':>8} {'s/q':>6} {'$/q':>10}")
for m in MODELS:
    r = res[m]
    lo, ln = r["look_ok"], max(r["look_n"], 1)
    ao, an = r["absent_ok"], max(r["absent_n"], 1)
    tot_ok, tot_n = lo + ao, r["look_n"] + r["absent_n"]
    cost = f"{spend[m]/max(calls[m],1):.5f}" if PRICE.get(m) else "  ?"
    print(f"{m:24} {f'{lo}/{ln}':>12} {f'{ao}/{an}':>10} "
          f"{f'{100*tot_ok/max(tot_n,1):.1f}%':>10} "
          f"{r['think']/max(calls[m],1):8.0f} {secs[m]/max(calls[m],1):6.1f} {cost:>10}")

print("\nmisses (question · expected · what it said):")
for m in MODELS:
    if not misses[m]:
        print(f"  {m}: none")
        continue
    seen = set()
    for q, exp, ans in misses[m]:
        if q in seen:
            continue
        seen.add(q)
        print(f"  {m}")
        print(f"    Q {q}   expected one of {exp}")
        print(f"    A {ans}")

# Questions ANY model missed. When they all miss the same one it is usually the
# question that is wrong, so the clean subset is reported separately.
contested = {q for m in MODELS for q, _, _ in misses[m]}
scored = [q for q in QUESTIONS if q["retrieved"] or q["kind"] == "absent"]
clean = [q for q in scored if q["q"] not in contested]
print(f"\nclean subset — {len(clean)} of {len(scored)} questions, "
      f"excluding every question ANY model missed:")
for m in MODELS:
    # Counted, not asserted: total attempts minus this model's misses that land
    # outside the contested set.
    off = sum(1 for q, _, _ in misses[m] if q not in contested)
    print(f"  {m:24} {len(clean)*TRIALS - off}/{len(clean)*TRIALS}")
print(f"  contested ({len(contested)}): " +
      "; ".join(sorted(q[:52] for q in contested)))

print(f"\nCAVEAT: one document ({os.path.basename(PDF)}), one layout. This ranks "
      f"models on THIS packet,\nnot in general. Set EVAL_PDF to widen it.")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "model_sweep_results.json"), "w", encoding="utf-8") as f:
    json.dump({"models": MODELS, "trials": TRIALS, "pdf": os.path.basename(PDF),
               "questions": [{k: v for k, v in q.items() if k != "ranked"}
                             for q in QUESTIONS],
               "scores": {m: dict(res[m]) for m in MODELS},
               "misses": {m: misses[m] for m in MODELS}},
              f, indent=2, default=str)
