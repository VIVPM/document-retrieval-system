"""
chunker.py — Text chunking strategies for logical documents.

Provides two chunking methods and a unified dispatcher:

  chunk_document_with_metadata()   — custom sliding-window (word-based)
  chunk_with_recursive_splitter()  — LangChain RecursiveCharacterTextSplitter
                                     (tiktoken / token-aware)
  process_all_documents()          — iterate logical docs → chunks
"""

from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.models import Block, ChunkMetadata, LogicalDocument


# ---------------------------------------------------------------------------
# Method 1: Custom sliding-window (word-based)
# ---------------------------------------------------------------------------

def chunk_document_with_metadata(
    logical_doc: LogicalDocument,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[ChunkMetadata]:
    """
    Chunk a logical document using a simple sliding window over words.

    The chunk_size and overlap are measured in *words*, not tokens.
    A stride of (chunk_size - overlap) words ensures adjacent chunks
    share `overlap` words of context.

    Args:
        logical_doc : source LogicalDocument
        chunk_size  : maximum words per chunk
        overlap     : words shared between consecutive chunks

    Returns:
        List of ChunkMetadata objects.
    """
    chunks_metadata: List[ChunkMetadata] = []
    words = logical_doc.text.split()

    if len(words) <= chunk_size:
        # Whole document fits in a single chunk
        chunks_metadata.append(ChunkMetadata(
            chunk_id=f"{logical_doc.doc_id}_chunk_0",
            doc_id=logical_doc.doc_id,
            doc_type=logical_doc.doc_type,
            filename=getattr(logical_doc, 'filename', 'unknown'),
            chunk_index=0,
            page_start=logical_doc.page_start,
            page_end=logical_doc.page_end,
            text=logical_doc.text,
        ))
        return chunks_metadata

    stride = chunk_size - overlap
    for i, start_idx in enumerate(range(0, len(words), stride)):
        end_idx = min(start_idx + chunk_size, len(words))
        chunk_text = " ".join(words[start_idx:end_idx])

        # Approximate page mapping
        chunk_position = start_idx / len(words)
        page_range = logical_doc.page_end - logical_doc.page_start
        relative_page = int(chunk_position * page_range)
        chunk_page_start = logical_doc.page_start + relative_page
        chunk_page_end = min(chunk_page_start + 1, logical_doc.page_end)

        chunks_metadata.append(ChunkMetadata(
            chunk_id=f"{logical_doc.doc_id}_chunk_{i}",
            doc_id=logical_doc.doc_id,
            doc_type=logical_doc.doc_type,
            filename=getattr(logical_doc, 'filename', 'unknown'),
            chunk_index=i,
            page_start=chunk_page_start,
            page_end=chunk_page_end,
            text=chunk_text,
        ))

        if end_idx >= len(words):
            break

    return chunks_metadata


# ---------------------------------------------------------------------------
# Method 3: Structure-aware — tables stay whole, prose is split by tokens
# ---------------------------------------------------------------------------

def _token_len(text: str) -> int:
    """Token count on the same encoder the splitter uses."""
    import tiktoken
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def _split_table(content: str, max_tokens: int) -> List[str]:
    """
    Split an oversized table by rows, repeating the header row in every part.

    Losing the header is the specific failure this whole strategy exists to
    prevent: `Origination | 2,150.00` without `Fee | Amount` above it is a
    number with no label, and a question about that field can no longer match
    it. So the header is duplicated rather than dropped.
    """
    rows = [r for r in content.split("\n") if r.strip()]
    if not rows:
        return []

    header, body = rows[0], rows[1:]
    if not body:
        return [header]

    parts: List[str] = []
    current: List[str] = []
    header_cost = _token_len(header)

    for row in body:
        candidate = current + [row]
        if current and header_cost + _token_len("\n".join(candidate)) > max_tokens:
            parts.append("\n".join([header] + current))
            current = [row]
        else:
            current = candidate

    if current:
        parts.append("\n".join([header] + current))
    return parts


def chunk_by_structure(
    logical_doc: LogicalDocument,
    chunk_size: int = 384,
    chunk_overlap: int = 48,
) -> List[ChunkMetadata]:
    """
    Chunk a document using the block types the extractor already identified.

    Two properties the token-only splitter cannot give:

    1. **A table is never severed from its header.** Each table becomes its own
       chunk; an oversized one is split by rows with the header repeated. The
       previous strategy flattened tables into the page text and hoped a "\\n"
       separator would land in the right place — but the splitter chooses
       boundaries by token budget, so a header and its data rows routinely
       ended up in different chunks. Nearly every question against this corpus
       is a table lookup, which is what makes that the dominant failure mode.

    2. **Page-accurate citations.** Blocks carry their page number, so a chunk
       is attributed to the pages it actually came from instead of inheriting
       the whole document's range. "Pages 3-3" instead of "Pages 0-11".

    Defaults are smaller than the old 512/128: with tables protected
    structurally, tighter prose chunks improve precision on value lookups, and
    the overlap no longer has to guard against severed tables.
    """
    if not logical_doc.blocks:
        # Nothing structured to work with (e.g. a doc built from text alone).
        return chunk_with_recursive_splitter(logical_doc, chunk_size, chunk_overlap)

    splitter = _recursive_splitter(chunk_size, chunk_overlap)

    # (text, page_start, page_end) triples, in reading order.
    pieces: List[Tuple[str, int, int]] = []
    prose: List[Block] = []

    def flush_prose():
        """Chunk the accumulated prose run and emit it."""
        if not prose:
            return
        joined = "\n\n".join(b.content for b in prose)
        lo = min(b.page_num for b in prose)
        hi = max(b.page_num for b in prose)
        for part in splitter.split_text(joined):
            if part.strip():
                pieces.append((part.strip(), lo, hi))
        prose.clear()

    for block in logical_doc.blocks:
        if block.kind == "table":
            # Emit any prose seen so far first, to preserve reading order.
            flush_prose()
            content = block.content.strip()
            if not content:
                continue
            if _token_len(content) <= chunk_size:
                pieces.append((content, block.page_num, block.page_num))
            else:
                for part in _split_table(content, chunk_size):
                    pieces.append((part, block.page_num, block.page_num))
        else:
            prose.append(block)

    flush_prose()

    return [
        ChunkMetadata(
            chunk_id=f"{logical_doc.doc_id}_chunk_{i}",
            doc_id=logical_doc.doc_id,
            doc_type=logical_doc.doc_type,
            filename=getattr(logical_doc, 'filename', 'unknown'),
            chunk_index=i,
            page_start=lo,
            page_end=hi,
            text=text,
        )
        for i, (text, lo, hi) in enumerate(pieces)
    ]


# ---------------------------------------------------------------------------
# Method 2: LangChain RecursiveCharacterTextSplitter (token-aware)
# ---------------------------------------------------------------------------

def _recursive_splitter(chunk_size: int, chunk_overlap: int):
    """Shared token-aware splitter, used for prose by both methods."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n\n",   # major section breaks
            "\n\n",     # paragraph breaks
            "\n",       # line breaks
            ". ",       # sentence endings
            ", ",       # clause breaks
            " ",        # word breaks
            "",         # character level (last resort)
        ],
        keep_separator=True,
    )


def chunk_with_recursive_splitter(
    logical_doc: LogicalDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> List[ChunkMetadata]:
    """
    Chunk using LangChain's RecursiveCharacterTextSplitter with tiktoken.

    Splits on a hierarchy of separators (triple newline → double newline →
    single newline → sentence → word → character) while respecting a
    token budget measured by the cl100k_base BPE tokeniser.

    This is the *default* method used by process_all_documents().

    Args:
        logical_doc   : source LogicalDocument
        chunk_size    : max tokens per chunk
        chunk_overlap : token overlap between adjacent chunks

    Returns:
        List of ChunkMetadata objects.
    """
    texts = _recursive_splitter(chunk_size, chunk_overlap).split_text(logical_doc.text)

    chunks_metadata: List[ChunkMetadata] = []
    for i, text in enumerate(texts):
        chunks_metadata.append(ChunkMetadata(
            chunk_id=f"{logical_doc.doc_id}_chunk_{i}",
            doc_id=logical_doc.doc_id,
            doc_type=logical_doc.doc_type,
            filename=getattr(logical_doc, 'filename', 'unknown'),
            chunk_index=i,
            page_start=logical_doc.page_start,
            page_end=logical_doc.page_end,
            text=text.strip(),
        ))

    return chunks_metadata


# ---------------------------------------------------------------------------
# Dispatcher: process all logical documents
# ---------------------------------------------------------------------------

CHUNKING_METHODS = {
    "structure":   chunk_by_structure,
    "recursive":   chunk_with_recursive_splitter,
    "sliding":     chunk_document_with_metadata,
}


def process_all_documents(
    logical_docs: List[LogicalDocument],
    chunking_method: str = "structure",
) -> List[ChunkMetadata]:
    """
    Chunk every logical document and return a flat list of all chunks.

    Args:
        logical_docs    : list of LogicalDocument objects
        chunking_method : one of 'structure' | 'recursive' | 'sliding'
                          (default: 'structure'). 'recursive' is kept so the
                          two can be compared on the same corpus — chunking
                          was never measured, so the default is a reasoned
                          choice, not a proven one.

    Returns:
        Flat list of ChunkMetadata covering all documents.
    """
    if chunking_method not in CHUNKING_METHODS:
        raise ValueError(
            f"Unknown chunking_method '{chunking_method}'. "
            f"Choose from: {list(CHUNKING_METHODS)}"
        )

    chunk_fn = CHUNKING_METHODS[chunking_method]
    all_chunks: List[ChunkMetadata] = []

    for logical_doc in logical_docs:
        chunks = chunk_fn(logical_doc)
        logical_doc.chunks = chunks
        all_chunks.extend(chunks)

        avg_tokens = sum(len(c.text.split()) for c in chunks) // max(len(chunks), 1)
        print(
            f"📄 {logical_doc.doc_type}: {len(chunks)} chunks "
            f"(avg ~{avg_tokens} tokens) [{chunking_method}]"
        )

    return all_chunks


# ---------------------------------------------------------------------------
# Self-check: python -m core.chunker
# ---------------------------------------------------------------------------

def _demo():
    """Assert the properties structure-aware chunking exists to guarantee."""
    doc = LogicalDocument(
        doc_id="doc_0", doc_type="Lender Fee Sheet",
        page_start=0, page_end=1, text="", filename="packet.pdf",
        blocks=[
            Block("text", "LOAN ESTIMATE\nDate Issued: 06/28/2011", 0),
            Block("table", "Fee | Amount | Paid By\n"
                           "Origination | 2,150.00 | Borrower\n"
                           "Appraisal | 575.00 | Borrower", 0),
            Block("text", "Closing Cost Details follow on the next page.", 1),
        ],
    )

    chunks = chunk_by_structure(doc, chunk_size=384, chunk_overlap=48)

    # 1. The table survives as exactly one chunk, header attached to its rows.
    tables = [c for c in chunks if "Origination | 2,150.00" in c.text]
    assert len(tables) == 1, f"table split across {len(tables)} chunks"
    assert "Fee | Amount | Paid By" in tables[0].text, "table lost its header"
    assert "LOAN ESTIMATE" not in tables[0].text, "table absorbed surrounding prose"

    # 2. Pages come from the blocks, not the document's whole range.
    assert tables[0].page_start == tables[0].page_end == 0, "wrong page for table"
    tail = [c for c in chunks if "next page" in c.text]
    assert tail and tail[0].page_start == 1, "prose page attribution wrong"

    # 3. An oversized table repeats the header in every part rather than
    #    orphaning rows from their labels.
    wide = "H1 | H2\n" + "\n".join(f"row{i} | val{i}" for i in range(400))
    parts = _split_table(wide, max_tokens=200)
    assert len(parts) > 1, "oversized table was not split"
    assert all(p.startswith("H1 | H2") for p in parts), "a part lost the header"
    assert sum(p.count("row") for p in parts) == 400, "rows lost while splitting"

    # 4. No blocks -> falls back rather than returning nothing.
    bare = LogicalDocument(doc_id="d", doc_type="Other", page_start=0, page_end=0,
                           text="plain text only, no blocks", filename="f.pdf")
    assert len(chunk_by_structure(bare)) == 1, "fallback path produced no chunks"

    print(f"chunker self-check OK — {len(chunks)} chunks, "
          f"{len(parts)} table parts")


if __name__ == "__main__":
    _demo()
