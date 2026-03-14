"""
answer_generator.py — LLM-based answer formulation with strict source attribution.

Formats retrieved chunks into a strict context block and prompts the
Gemma-2 LLM to answer the user's query using *only* that context.
"""

from typing import List, Tuple, Dict
from core.models import ChunkMetadata
from llm.llm_router import llm as gemma_llm

def generate_answer_with_sources(
    query: str,
    retrieved_chunks: List[Tuple[ChunkMetadata, float]]
) -> Dict:
    """
    Generate an answer using retrieved chunks as context.

    Args:
        query            : the user's question
        retrieved_chunks : list of (ChunkMetadata, score) tuples

    Returns:
        Dict containing:
          - 'answer'     : LLM generated string
          - 'sources'    : list of metadata dicts for UI display
          - 'confidence' : avg retrieval score
          - 'chunks_used': number of chunks included
    """
    if not retrieved_chunks:
        return {
            'answer': "I couldn't find relevant information to answer your question.",
            'sources': [],
            'confidence': 0.0,
            'chunks_used': 0
        }

    context_parts = []
    sources = []

    for chunk_meta, score in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk_meta.doc_type} | "
            f"Pages: {chunk_meta.page_start}-{chunk_meta.page_end}]"
        )
        context_parts.append(chunk_meta.text)
        context_parts.append("")

        sources.append({
            'doc_type': chunk_meta.doc_type,
            'pages': f"{chunk_meta.page_start}-{chunk_meta.page_end}",
            'relevance': f"{score:.2%}",
            'preview': chunk_meta.text
        })

    context = "\n".join(context_parts)

    prompt = f"""You are a financial document question-answering assistant.
      Your job is to answer the user's question accurately using ONLY the provided context.

      Context:
      {context}

      Question: {query}

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

      Answer:"""

    try:
        response = gemma_llm.complete(prompt)
        answer = response.text.strip()

        non_zero_scores = [s for _, s in retrieved_chunks if s > 0]
        avg_score = sum(non_zero_scores) / len(non_zero_scores) if non_zero_scores else 0.0

        return {
            'answer': answer,
            'sources': sources,
            'confidence': avg_score,
            'chunks_used': len(retrieved_chunks)
        }
    except Exception as e:
        return {
            'answer': f"Error generating answer: {str(e)}",
            'sources': sources,
            'confidence': 0.0,
            'chunks_used': 0
        }
