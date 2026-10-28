"""Bounded source context shared by retrieval and answer generation."""

from core.chunker import _token_len

MAX_EVIDENCE_CHUNKS = 18
MAX_EVIDENCE_TOKENS = 6000


def source_block(chunk):
    return (f"[Source: {chunk.filename} | {chunk.doc_type} | "
            f"Document: {chunk.doc_id} | "
            f"Pages: {chunk.page_start + 1}-{chunk.page_end + 1}]\n"
            f"{chunk.text}\n\n")


def bounded_evidence(candidates):
    """Keep whole chunks in priority order; never cut a table to fit the budget."""
    selected = []
    seen_ids = set()
    seen_text = set()
    tokens = 0
    for chunk, score in candidates:
        text_key = (chunk.doc_id, chunk.page_start, chunk.page_end,
                    " ".join(chunk.text.split()))
        if not chunk.text.strip() or chunk.chunk_id in seen_ids or text_key in seen_text:
            continue
        cost = _token_len(source_block(chunk))
        if tokens + cost > MAX_EVIDENCE_TOKENS:
            continue
        selected.append((chunk, score))
        seen_ids.add(chunk.chunk_id)
        seen_text.add(text_key)
        tokens += cost
        if len(selected) >= MAX_EVIDENCE_CHUNKS:
            break
    return selected, tokens
