"""
chunker.py — Text chunking strategies for logical documents.

Provides two chunking methods and a unified dispatcher:

  chunk_document_with_metadata()   — custom sliding-window (word-based)
  chunk_with_recursive_splitter()  — LangChain RecursiveCharacterTextSplitter
                                     (tiktoken / token-aware)
  process_all_documents()          — iterate logical docs → chunks
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.models import ChunkMetadata, LogicalDocument


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
# Method 2: LangChain RecursiveCharacterTextSplitter (token-aware)
# ---------------------------------------------------------------------------

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
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n\n",   # major section breaks
            "\n\n",     # paragraph breaks
            "\n",       # line breaks (important for tables)
            ". ",       # sentence endings
            ", ",       # clause breaks
            " ",        # word breaks
            "",         # character level (last resort)
        ],
        keep_separator=True,
    )

    texts = splitter.split_text(logical_doc.text)

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
    "recursive":   chunk_with_recursive_splitter,
    "sliding":     chunk_document_with_metadata,
}


def process_all_documents(
    logical_docs: List[LogicalDocument],
    chunking_method: str = "recursive",
) -> List[ChunkMetadata]:
    """
    Chunk every logical document and return a flat list of all chunks.

    Args:
        logical_docs    : list of LogicalDocument objects
        chunking_method : one of 'recursive' | 'sliding'
                          (default: 'recursive')

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
