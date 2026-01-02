"""
models.py — Core data structures for the RAG pipeline.

Defines all dataclasses used across the system:
  - PageInfo        : a single extracted page from a PDF
  - LogicalDocument : a detected logical document within a multi-doc PDF
  - ChunkMetadata   : a text chunk with rich provenance metadata
  - SearchConfig    : resolved configuration for a retrieval operation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Page-level representation
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """
    One layout element as Docling found it, in reading order.

    `kind` is the whole point: Docling already knows what is a table and what
    is prose, and the chunker needs that to avoid severing a table from its
    header row. Carrying `page_num` per block is what makes chunk citations
    page-accurate instead of spanning the whole document's range.
    """
    kind: str        # "text" | "table"
    content: str
    page_num: int    # 0-indexed


@dataclass
class PageInfo:
    """Stores information about a single page extracted from a PDF."""
    page_num: int
    text: str
    doc_type: Optional[str] = None
    page_in_doc: int = 0
    # Same content as `text`, but structured. `text` is kept for the
    # classifier and boundary detector, which want a flat string.
    blocks: List[Block] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Logical document (may span multiple pages)
# ---------------------------------------------------------------------------

@dataclass
class LogicalDocument:
    """
    Represents one logical document detected within a (potentially
    multi-document) PDF file.
    """
    doc_id: str
    doc_type: str
    page_start: int
    page_end: int
    text: str
    filename: Optional[str] = field(default=None)
    chunks: Optional[List[Dict]] = field(default=None)
    # Blocks of every page in this document, in reading order. The chunker
    # prefers these over `text`; it falls back to `text` when empty.
    blocks: List[Block] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Chunk with provenance metadata
# ---------------------------------------------------------------------------

@dataclass
class ChunkMetadata:
    """
    A single text chunk with rich metadata for retrieval and attribution.

    Attributes:
        chunk_id    : unique identifier (<doc_id>_chunk_<index>)
        doc_id      : parent document identifier
        doc_type    : document category (e.g. "Bank Statement")
        filename    : name of the source PDF file
        chunk_index : position of this chunk within the document
        page_start  : first PDF page this chunk originates from
        page_end    : last PDF page this chunk originates from
        text        : raw chunk text
    """
    chunk_id: str
    doc_id: str
    doc_type: str
    filename: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str


# ---------------------------------------------------------------------------
# Resolved search configuration (produced by the retriever's routing logic)
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    """Configuration resolved for a single search operation."""
    use_filtered: bool
    selected_type: Optional[str]
    total_chunks: int
    routing_info: Dict
