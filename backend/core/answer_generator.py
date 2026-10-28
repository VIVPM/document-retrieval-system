"""Answer formulation with strict source attribution."""

from typing import List, Tuple, Dict
from core.models import ChunkMetadata
from core.evidence import source_block
from llm.llm_router import llm as gemma_llm

SYSTEM_RULES = """You are a financial document question-answering assistant.
Your job is to answer the user's question accurately using ONLY the provided context.

INSTRUCTIONS:
1. Answer the question directly and concisely.
2. Source citations are rendered separately by the application. Do not repeat
   source citations, page references, Document IDs, filenames or a "Source:"
   section in the answer body, even when the question asks for citations.
   You may name a document type when needed to explain a comparison.
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
  their field labels, people or periods. A wrong answer is worse than no answer.

8. CALCULATIONS (when requested or necessary to explain a comparison or discrepancy):
  - Verify each input and its source location internally.
  - Compute and double-check the arithmetic yourself.
  - Show the relevant inputs, equation and result concisely, without source
    metadata or a long step-by-step checklist. Do not use LaTeX delimiters.

9. LINKED EVIDENCE: Chunks with the same Document ID are fragments of one
   document, not independent payments or income sources. Connect facts only
   when the borrower, employer/account, dates and pay periods support the link.
   Do not combine one person's earnings with another person's pay frequency,
   add year-to-date earnings to current pay, or count overlapping rows twice.
   Before calculating, identify the person and employer/account for EACH
   Document ID used. An anonymous amount in a different Document ID cannot
   inherit a name or pay frequency from another document, even if their
   document types match. If that document's identity is missing, abstain
   from attributing its amount to the borrower.
   If a required identity or pay frequency is missing, say what is unknown.
   A name in a transaction description is not proof of account ownership;
   an unexplained deposit needs verification, not a claim of contradiction.
   Do not call a transfer third-party funds, a gift or a loan unless the
   evidence establishes that classification. Retrieved excerpts are not a
   complete inventory: say a document is not in the provided context rather
   than declaring it absent from the entire packet.

"""

SUMMARY_RULES = """You are a financial document assistant. Summarize the document
below for a loan officer, using ONLY the provided context.

- Name each distinct document in the packet and what it contains.
- Report key figures (amounts, rates, dates, parties) exactly as they appear;
  do not round, reformat, or estimate.
- Source citations are rendered separately by the application. Do not repeat
  source citations, page references, Document IDs, filenames or a "Source:"
  section in the answer body, even when the question asks for citations.
- Do not add information that is not in the context.

"""

NO_CONTEXT_ANSWER = "I couldn't find relevant information to answer your question."
LLM_EMPTY_ANSWER = ("The language model did not return an answer. The sources "
                    "below were retrieved successfully — please retry, or "
                    "rephrase the question.")


def build_sources(retrieved_chunks: List[Tuple[ChunkMetadata, float]],
                  k: int = 6) -> List[Dict]:
    """The source list the UI renders: the top k search matches only. Linked
    neighbour chunks (score 0) still reach the prompt but are not shown."""
    matches = [(c, score) for c, score in retrieved_chunks if score][:k]
    return [
        {
            'filename': c.filename,
            'doc_type': c.doc_type,
            'pages': f"{c.page_start + 1}-{c.page_end + 1}",
            'relevance': f"{score:.2%}",
            'preview': c.text,
        }
        for c, score in matches
    ]


def build_prompt(query: str, retrieved_chunks: List[Tuple[ChunkMetadata, float]],
                 summarize: bool = False) -> str:
    """The exact prompt sent to the model. Shared by the sync and streaming
    paths so the two cannot drift — rules first, context second (see the
    module docstring)."""
    context = "".join(source_block(c) for c, _ in retrieved_chunks)
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
