"""Turns a follow-up question into a standalone one."""

from typing import Dict, List

from llm.llm_router import llm

MAX_HISTORY_MESSAGES = 6

_PROMPT = """You rewrite follow-up questions about a single document so they can \
be understood on their own.

The rewritten question is used to SEARCH the document, so it must contain the \
actual subject being asked about — not a pronoun or an implied reference.

RULES
1. Resolve a pronoun (it, that, they, this, those) ONLY when its referent is in
   the HISTORY. If the pronoun refers to something already named in the LATEST
   QUESTION itself, leave the question completely alone — it is already
   self-contained.
2. Resolve ellipsis — a question missing its subject or its field. "And the
   appraisal?" after a question about fees becomes "What is the appraisal fee
   amount?".
3. If the LATEST QUESTION is ALREADY standalone and unambiguous, return it
   EXACTLY as written, character for character. Do not improve it, expand it or
   rephrase it.
4. If the LATEST QUESTION is about the conversation itself rather than the
   document ("what did I ask before?", "summarise our chat", "repeat that"),
   return it EXACTLY as written.
5. On a topic change, follow the NEW topic. Do not carry the old subject over.
6. Keep the user's own wording and any exact field labels or figures they used —
   those drive lexical (BM25) matching and must not be paraphrased.
7. Output ONLY the rewritten question. No quotes, no preamble, no explanation.

EXAMPLES

HISTORY:
User: What interest rate is being offered?
Assistant: 4.25%, locked until 07/28/2011.
LATEST QUESTION: and when does it lock?
OUTPUT: When does the 4.25% interest rate lock expire?

HISTORY:
User: What are the origination charges?
Assistant: $2,150.00 on the Lender Fee Sheet.
LATEST QUESTION: and the appraisal?
OUTPUT: What is the appraisal fee amount?

HISTORY:
User: What is the loan amount?
Assistant: $380,000.
LATEST QUESTION: What is the net pay on the pay slip?
OUTPUT: What is the net pay on the pay slip?

HISTORY:
User: List the closing costs.
Assistant: Underwriting $550.00, Wire Transfer $75.00, ...
LATEST QUESTION: which of those is the largest?
OUTPUT: Which of the closing costs is the largest: underwriting, wire transfer, \
administration or appraisal?

HISTORY:
User: What is the interest rate?
Assistant: 4.25%.
LATEST QUESTION: what was my first question?
OUTPUT: what was my first question?

HISTORY:
User: What is the interest rate?
Assistant: 4.25%.
LATEST QUESTION: Who pays the Title Insurance and how much is it?
OUTPUT: Who pays the Title Insurance and how much is it?
"""


def _format_history(messages: List[Dict]) -> str:
    lines = []
    for m in messages[-MAX_HISTORY_MESSAGES:]:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = (m.get("content") or "").strip().replace("\n", " ")
        if role == "Assistant" and len(content) > 300:
            content = content[:300] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def rewrite_standalone(question: str, history: List[Dict]) -> str:
    """    Rewrite `question` into a self-contained query using recent turns."""
    if not history:
        return question

    prompt = (f"{_PROMPT}\n\nHISTORY:\n{_format_history(history)}\n"
              f"LATEST QUESTION: {question}\nOUTPUT:")

    try:
        rewritten = llm.complete(
            prompt, temperature=0.0, max_tokens=256, thinking_budget=0
        ).text.strip()
    except Exception as e:
        print(f"⚠️ Query rewrite failed ({type(e).__name__}: {e}) — using the raw question")
        return question

    rewritten = rewritten.strip('"').strip("'").strip()
    if not rewritten or len(rewritten) > 4 * max(len(question), 60):
        print("⚠️ Query rewrite looked wrong — using the raw question")
        return question

    if rewritten != question:
        print(f"✍️  Rewrote query: {question!r} → {rewritten!r}")
    return rewritten
