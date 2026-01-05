"""
A/B tests a change to the answer prompt.

Four groups, because loosening a rule is easy to "win" by breaking it:
RESCUE (must answer a differently-worded label), NOREFUSE (must not decline
something answerable), DISAMBIG (must pick the right one of two similar fields
and not the decoy), REFUSE (must still decline when the value is absent).

Answers run at temperature 0.3, so each question runs EVAL_TRIALS times.
"""
import math
import os
import re
import sys
import time
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(BACKEND)
PDF = os.getenv("EVAL_PDF", os.path.join(REPO, "Test Blob File.pdf"))
sys.path.insert(0, BACKEND)

from dotenv import load_dotenv
load_dotenv(os.path.join(BACKEND, ".env"))

import core.answer_generator as ag
from core.chunker import chunk_by_structure
from core.pdf_processor import extract_and_analyze_pdf
from llm.llm_router import embed_model

MODEL = os.getenv("EVAL_MODEL", "gemini-2.5-flash")
TRIALS = int(os.getenv("EVAL_TRIALS", "3"))
K = 6

REFUSAL = re.compile(
    r"does not contain|doesn't contain|not contain|no information|not (?:be )?found|"
    r"cannot determine|can't determine|not (?:explicitly )?(?:present|available|stated|provided)|"
    r"unable to (?:answer|determine)|not enough information|insufficient",
    re.I)

# (group, question, must-appear, must-NOT-appear)
CASES = [
    # ── The document labels it differently from the question. Answerable.
    ("RESCUE", "What is the Loan Amount?",            ["380,000"], []),
    ("RESCUE", "What are the closing costs?",         ["4,520"],   []),
    ("RESCUE", "What is the appraisal cost?",         ["525"],     []),
    ("RESCUE", "What is the credit report charge?",   ["25"],      []),
    ("RESCUE", "How much is the underwriting?",       ["550"],     []),

    # The phrasings observed refusing, verbatim. Answerable under a fair
    # reading, so the only requirement is that the model does not decline.
    ("NOREFUSE", "List the closing costs and say which is the largest.", [], []),
    ("NOREFUSE", "What is the Loan Amount?", [], []),
    ("NOREFUSE", "What is the monthly payment?", [], []),

    # ── Similar labels, different values. Rule 4's whole reason to exist.
    ("DISAMBIG", "What is the Total Monthly Payment?",
     ["2,308.95"], ["1,869.37"]),
    ("DISAMBIG", "What is the Principal & Interest payment?",
     ["1,869.37"], ["2,308.95"]),
    ("DISAMBIG", "What is the Purchase Price?",
     ["475,000"],  ["380,000"]),
    ("DISAMBIG", "What is the Total Estimated Funds needed to close?",
     ["95,641.53"], ["4,520"]),
    ("DISAMBIG", "What is the Est. Prepaid Items/Reserves amount?",
     ["1,121.53"], ["4,520"]),

    # ── Genuinely absent. Softening must not turn these into guesses.
    ("REFUSE", "What is the Annual Percentage Rate (APR)?", [], []),
    ("REFUSE", "What is the borrower's social security number?", [], []),
    ("REFUSE", "When does the interest rate lock expire?", [], []),
]

OLD_RULE4 = """4. FIELD MATCHING: Identify the EXACT field label mentioned in the question.
  - "Total Loan Costs" is NOT "Total Closing Costs"
  - "Monthly Payment" is NOT "Total Monthly Payment"
  - "Loan Amount" is NOT "Sale Price"
  - Find the EXACT label first, then extract the value next to it."""

# Separates "the document words the same field differently" (answer it) from
# "this is a different field that looks similar" (never substitute).
NEW_RULE4 = """4. FIELD MATCHING: Find the field the question is asking about.
  - Prefer an exact label match.
  - If the document words the SAME field differently — "Total Loan Amount" for
    "Loan Amount", "Appraisal Fee" for "appraisal cost", "Est. Closing Costs"
    for "closing costs" — use that row and state which label you used.
  - NEVER substitute a DIFFERENT field that merely looks similar. These name
    distinct fields, not wording variants:
      "Total Loan Costs" is NOT "Total Closing Costs"
      "Monthly Payment" is NOT "Total Monthly Payment"
      "Loan Amount" is NOT "Sale Price" or "Purchase Price"
  - If two rows both plausibly answer the question and you cannot tell which is
    meant, do not choose — follow rule 7 and list both.
  - Only say the document does not contain the field if NO row plausibly names
    it. A label that differs only in wording is not a missing field."""


def cosine(u, v):
    dot = sum(x * y for x, y in zip(u, v))
    return dot / (math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(y * y for y in v)) + 1e-12)


def verdict(group, ans, expect, forbid):
    """True = behaved correctly for its group."""
    if group == "REFUSE":
        return bool(REFUSAL.search(ans))
    if group == "NOREFUSE":
        return not REFUSAL.search(ans)
    if REFUSAL.search(ans) and not any(e in ans for e in expect):
        return False                      # refused something answerable
    if not any(e in ans for e in expect):
        return False
    return not any(f in ans for f in forbid)


print(f"[extract] {os.path.basename(PDF)}")
pages, docs = extract_and_analyze_pdf(PDF, filename=os.path.basename(PDF))
chunks = [c for d in docs for c in chunk_by_structure(d)]
print(f"[chunk] {len(chunks)} chunks")

cvecs = embed_model.encode([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")
qvecs = embed_model.encode([q for _, q, _, _ in CASES], task_type="RETRIEVAL_QUERY")

VARIANTS = {"old": ag.SYSTEM_RULES,
            "new": ag.SYSTEM_RULES.replace(OLD_RULE4, NEW_RULE4)}
assert VARIANTS["new"] != VARIANTS["old"], "rule 4 text not found — prompt drifted"

score = {v: defaultdict(lambda: [0, 0]) for v in VARIANTS}   # group -> [ok, n]
detail = []

for i, (group, question, expect, forbid) in enumerate(CASES):
    ranked = sorted(zip(chunks, (cosine(qvecs[i], cv) for cv in cvecs)),
                    key=lambda t: t[1], reverse=True)[:K]
    # A RESCUE case is only meaningful if the value was actually retrieved.
    ctx = "\n".join(c.text for c, _ in ranked)
    retrievable = not expect or any(e in ctx for e in expect)

    line = {"group": group, "q": question, "retrievable": retrievable}
    print(f"\n[{group}] {question}" + ("" if retrievable else "   ⚠️ value NOT retrieved"))

    for name, rules in VARIANTS.items():
        ag.SYSTEM_RULES = rules
        ok = 0
        sample = ""
        for _ in range(TRIALS):
            ans = ag.generate_answer_with_sources(question, ranked, model=MODEL)["answer"]
            ok += verdict(group, ans, expect, forbid)
            sample = sample or ans
        ag.SYSTEM_RULES = VARIANTS["old"]

        if retrievable or group in ("REFUSE", "NOREFUSE"):
            score[name][group][0] += ok
            score[name][group][1] += TRIALS
        line[name] = ok
        print(f"   {name}: {ok}/{TRIALS}   {sample[:100].replace(chr(10),' ')}")
    detail.append(line)

print(f"\n{'='*70}\nPASS RATE by group ({TRIALS} trials/question, {MODEL})\n")
print(f"{'group':10} {'old':>12} {'new':>12}   what it protects")
blurb = {"RESCUE": "answers a differently-worded label",
         "NOREFUSE": "does not decline something answerable",
         "DISAMBIG": "picks the right one of two similar fields",
         "REFUSE": "still declines when genuinely absent"}
for g in ("RESCUE", "NOREFUSE", "DISAMBIG", "REFUSE"):
    o, n = score["old"][g], score["new"][g]
    print(f"{g:10} {f'{o[0]}/{o[1]}':>12} {f'{n[0]}/{n[1]}':>12}   {blurb[g]}")

print("\nregressions (new worse than old on a question):")
bad = [d for d in detail if d["new"] < d["old"]]
for d in bad:
    print(f"  [{d['group']}] {d['q']}  old={d['old']} new={d['new']}")
if not bad:
    print("  none")
