"""models.py — Core data structures for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Block:
    """    One layout element as the extractor found it, in reading order."""
    kind: str
    content: str
    page_num: int


@dataclass
class PageInfo:
    """Stores information about a single page extracted from a PDF."""
    page_num: int
    text: str
    doc_type: Optional[str] = None
    page_in_doc: int = 0
    blocks: List[Block] = field(default_factory=list)
    kv: List[tuple] = field(default_factory=list)
    page_type: Optional[str] = None


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
    blocks: List[Block] = field(default_factory=list)
    kv: List[tuple] = field(default_factory=list)
    page_types: List[str] = field(default_factory=list)


@dataclass
class ChunkMetadata:
    """    A single text chunk with rich metadata for retrieval and attribution."""
    chunk_id: str
    doc_id: str
    doc_type: str
    filename: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    context: Optional[str] = None


@dataclass
class SearchConfig:
    """Configuration resolved for a single search operation."""
    use_filtered: bool
    selected_type: Optional[str]
    total_chunks: int
    routing_info: Dict
