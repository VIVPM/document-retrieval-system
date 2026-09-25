"""Answer formulation with strict source attribution."""

from typing import List, Tuple, Dict
from core.models import ChunkMetadata
from llm.llm_router import llm as gemma_llm

SYSTEM_RULES = """You are a financial document question-answering assistant.
Your job is to answer the user's question accurately using ONLY the provided context.

INSTRUCTIONS:
1. Answer the question directly and concisely.
2. Cite which document type and page number your answer comes from.
3. If the context doesn't contain enough information to answer, say so clearly.

STRICT RULES FOR ACCURACY:
4. FIELD MATCHING: Identify the EXACT field label mentioned in the question.
  - "Total Loan Costs" is NOT "Total Closing Costs"
  - "Monthly Payment" is NOT "Total Monthly Payment"
  - "Loan Amount" is NOT "Sale Price"
  - Find the EXACT label first, then extract the value next to it.

5. DOCUMENT MATCHING: If the question mentions a specific document
  (e.g., "on the Loan Estimate", "on the Payworks payslip"),
  ONLY use values from that specific document's chunks.
  Ignore values from other documents even if they look similar.

6. EXACT VALUES: Report numbers exactly as they appear in the text.
  - Use digits not words (write "7" not "seven").
  - Keep original date formats (write "06/28/2011" not "June 28, 2011").
  - Include currency symbols and units as they appear.
  - Do not round, reformat, or estimate.

7. DO NOT GUESS: If you find multiple similar values and cannot determine
  which one answers the question, say so and list the candidates with
  their source locations. A wrong answer is worse than no answer.

8. CALCULATIONS (only when the question asks to compute something):
  - Step 1: List each value and its exact source location.
  - Step 2: Write the mathematical equation.
  - Step 3: Compute step by step, showing your work.
  - Step 4: Double-check by recalculating.
  - Step 5: State the final answer.

"""

SUMMARY_RULES = """You are a financial document assistant. Summarize the document
below for a loan officer, using ONLY the provided context.

- Name each distinct document in the packet and what it contains.
- Report key figures (amounts, rates, dates, parties) exactly as they appear;
  do not round, reformat, or estimate.
- Cite the document type and page for each figure.
- Do not add information that is not in the context.

"""

NO_CONTEXT_ANSWER = "I couldn't find relevant information to answer your question."
LLM_EMPTY_ANSWER = ("The language model did not return an answer. The sources "
                    "below were retrieved successfully — please retry, or "
                    "rephrase the question.")


def build_sources(retrieved_chunks: List[Tuple[ChunkMetadata, float]]) -> List[Dict]:
    """The source list the UI renders, one entry per retrieved chunk."""
    return [
        {
            'filename': c.filename,
            'doc_type': c.doc_type,
            'pages': f"{c.page_start + 1}-{c.page_end + 1}",
            'relevance': f"{score:.2%}",
            'preview': c.text,
        }
        for c, score in retrieved_chunks
    ]


def build_prompt(query: str, retrieved_chunks: List[Tuple[ChunkMetadata, float]],
                 summarize: bool = False) -> str:
    """The exact prompt sent to the model. Shared by the sync and streaming
    paths so the two cannot drift — rules first, context second (see the
    module docstring)."""
    context_parts = []
    for c, _ in retrieved_chunks:
        context_parts.append(
            f"[Source: {c.filename} | {c.doc_type} | "
            f"Pages: {c.page_start + 1}-{c.page_end + 1}]"
        )
        context_parts.append(c.text)
        context_parts.append("")
    context = "\n".join(context_parts)
    rules = SUMMARY_RULES if summarize else SYSTEM_RULES
    return f"""{rules}
Context:
{context}

Question: {query}

Answer:"""


def stream_answer(query: str, retrieved_chunks: List[Tuple[ChunkMetadata, float]],
                  summarize: bool = False):
    """Yield answer tokens for the SSE path. Empty retrieval yields the same
    canned line generate_answer_with_sources returns, and no model is called."""
    if not retrieved_chunks:
        yield NO_CONTEXT_ANSWER
        return
    yield from gemma_llm.stream(build_prompt(query, retrieved_chunks, summarize))


def generate_answer_with_sources(
    query: str,
    retrieved_chunks: List[Tuple[ChunkMetadata, float]],
    model: str | None = None,
    thinking_budget: int | None = None,
) -> Dict:
    """    Answer `query` from `retrieved_chunks` alone."""
    if not retrieved_chunks:
        return {
            'answer': NO_CONTEXT_ANSWER,
            'sources': [],
            'confidence': 0.0,
            'chunks_used': 0
        }

    sources = build_sources(retrieved_chunks)
    prompt = build_prompt(query, retrieved_chunks)

    try:
        kw = {} if thinking_budget is None else {"thinking_budget": thinking_budget}
        response = gemma_llm.complete(prompt, model=model, **kw)
        answer = response.text.strip()

        if not answer:
            return {
                'answer': LLM_EMPTY_ANSWER,
                'sources': sources,
                'confidence': 0.0,
                'chunks_used': len(retrieved_chunks),
                'llm_failed': True,
            }

        non_zero_scores = [s for _, s in retrieved_chunks if s > 0]
        avg_score = sum(non_zero_scores) / len(non_zero_scores) if non_zero_scores else 0.0

        return {
            'answer': answer,
            'sources': sources,
            'confidence': avg_score,
            'chunks_used': len(retrieved_chunks),
            'usage': response.usage,
        }
    except Exception as e:
        return {
            'answer': f"Error generating answer: {str(e)}",
            'sources': sources,
            'confidence': 0.0,
            'chunks_used': 0
        }
