"""chunker.py — Text chunking strategies for logical documents."""

from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.models import Block, ChunkMetadata, LogicalDocument


def chunk_document_with_metadata(
    logical_doc: LogicalDocument,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[ChunkMetadata]:
    """    Chunk a logical document using a simple sliding window over words."""
    chunks_metadata: List[ChunkMetadata] = []
    words = logical_doc.text.split()

    if len(words) <= chunk_size:
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


def _token_len(text: str) -> int:
    """Token count on the same encoder the splitter uses."""
    import tiktoken
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def _split_table(content: str, max_tokens: int) -> List[str]:
    """    Split an oversized table by rows, repeating the header row in every part."""
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
    """    Chunk a document using the block types the extractor already identified."""
    if not logical_doc.blocks:
        return chunk_with_recursive_splitter(logical_doc, chunk_size, chunk_overlap)

    splitter = _recursive_splitter(chunk_size, chunk_overlap)

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


def _recursive_splitter(chunk_size: int, chunk_overlap: int):
    """Shared token-aware splitter, used for prose by both methods."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n\n",
            "\n\n",
            "\n",
            ". ",
            ", ",
            " ",
            "",
        ],
        keep_separator=True,
    )


def chunk_with_recursive_splitter(
    logical_doc: LogicalDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> List[ChunkMetadata]:
    """    Chunk using LangChain's RecursiveCharacterTextSplitter with tiktoken."""
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


CHUNKING_METHODS = {
    "structure":   chunk_by_structure,
    "recursive":   chunk_with_recursive_splitter,
    "sliding":     chunk_document_with_metadata,
}


def process_all_documents(
    logical_docs: List[LogicalDocument],
    chunking_method: str = "structure",
) -> List[ChunkMetadata]:
    """    Chunk every logical document and return a flat list of all chunks."""
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

    tables = [c for c in chunks if "Origination | 2,150.00" in c.text]
    assert len(tables) == 1, f"table split across {len(tables)} chunks"
    assert "Fee | Amount | Paid By" in tables[0].text, "table lost its header"
    assert "LOAN ESTIMATE" not in tables[0].text, "table absorbed surrounding prose"

    assert tables[0].page_start == tables[0].page_end == 0, "wrong page for table"
    tail = [c for c in chunks if "next page" in c.text]
    assert tail and tail[0].page_start == 1, "prose page attribution wrong"

    wide = "H1 | H2\n" + "\n".join(f"row{i} | val{i}" for i in range(400))
    parts = _split_table(wide, max_tokens=200)
    assert len(parts) > 1, "oversized table was not split"
    assert all(p.startswith("H1 | H2") for p in parts), "a part lost the header"
    assert sum(p.count("row") for p in parts) == 400, "rows lost while splitting"

    bare = LogicalDocument(doc_id="d", doc_type="Other", page_start=0, page_end=0,
                           text="plain text only, no blocks", filename="f.pdf")
    assert len(chunk_by_structure(bare)) == 1, "fallback path produced no chunks"

    print(f"chunker self-check OK — {len(chunks)} chunks, "
          f"{len(parts)} table parts")


if __name__ == "__main__":
    _demo()
