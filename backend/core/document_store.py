"""
document_store.py — The main orchestration layer.

Glues together PDF processing, chunking, retrieval, and generation into
a single stateful `EnhancedDocumentStoreHybrid` class that can be cleanly
used by the UI.
"""

from datetime import datetime
from typing import List, Dict, Optional

from core.pdf_processor import extract_and_analyze_pdf
from core.chunker import process_all_documents
from core.retriever import HybridRetriever
from core.answer_generator import generate_answer_with_sources


class EnhancedDocumentStoreHybrid:
    """
    Manages the complete document processing and retrieval pipeline.
    Uses hybrid search (Pinecone + BM25) with RRF and optional reranking.
    """

    def __init__(self, use_rerank: bool = False):
        self.pages_info = []
        self.logical_docs = []
        self.chunks_metadata = []

        self.retriever = HybridRetriever(rrf_k=60, use_rerank=use_rerank)
        self.use_rerank = use_rerank

        self.is_ready = False
        self.processing_stats = {}
        self.filename = None

    def clear(self):
        """Clear the current document data and indices."""
        self.pages_info = []
        self.logical_docs = []
        self.chunks_metadata = []
        self.retriever.clear()
        self.is_ready = False
        self.processing_stats = {}
        self.filename = None

    def process_pdf(self, pdf_file, filename: str = "document.pdf", embed_model=None) -> tuple[bool, dict]:
        """
        Run the complete ingestion pipeline:
          Docling extraction → Classifier boundaries → Chunker → Retriever mapping
        """
        self.filename = filename
        self.is_ready = False
        start_time = datetime.now()

        try:
            self.pages_info, self.logical_docs = extract_and_analyze_pdf(pdf_file, filename)
            self.chunks_metadata = process_all_documents(self.logical_docs)
            self.retriever.build_indices(self.chunks_metadata, embed_model)

            process_time = (datetime.now() - start_time).total_seconds()
            self.processing_stats = {
                'filename': filename,
                'total_pages': len(self.pages_info),
                'documents_found': len(self.logical_docs),
                'total_chunks': len(self.chunks_metadata),
                'document_types': list(set(doc.doc_type for doc in self.logical_docs)),
                'processing_time': f"{process_time:.1f}s",
                'search_type': self._get_search_type_label(),
                'rerank_enabled': self.use_rerank
            }

            self.is_ready = True
            return True, self.processing_stats

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, {'error': str(e)}

    def query(
        self, question: str, filter_type: Optional[str] = None,
        auto_route: bool = False, k: int = 4, return_details: bool = False
    ) -> Dict:
        """Query the document store and generate an answer."""
        if not self.is_ready:
            return {
                'answer': "Please upload and process a PDF first.",
                'sources': [],
                'confidence': 0.0
            }

        retrieval_result = self.retriever.retrieve(
            question,
            k=k,
            filter_doc_type=filter_type,
            auto_route=auto_route,
            return_details=return_details
        )
        
        # Determine how to unpack retriever output 
        # Since retriever returns return_details dict or list format depending on args
        if return_details:
            retrieval_list = retrieval_result.get('results', [])
        else:
            retrieval_list = retrieval_result

        retrieved = [(r['chunk'], r['final_score']) for r in retrieval_list]

        if return_details:
            result = generate_answer_with_sources(question, retrieved)
            result['retrieval_details'] = retrieval_result
        else:
            result = generate_answer_with_sources(question, retrieved)

        result['filter_used'] = filter_type or ('auto' if auto_route else 'none')
        return result

    def set_rerank(self, use_rerank: bool):
        """Enable or disable cross-encoder reranking interactively."""
        self.retriever.set_rerank(use_rerank)
        self.use_rerank = self.retriever.use_rerank

        if self.processing_stats:
            self.processing_stats['rerank_enabled'] = self.use_rerank
            self.processing_stats['search_type'] = self._get_search_type_label()

    def get_document_structure(self) -> List[Dict]:
        """Summarise the ingested documents for the UI dashboard."""
        if not self.logical_docs:
            return []
        return [
            {
                'id': doc.doc_id,
                'type': doc.doc_type,
                'pages': f"{doc.page_start + 1}-{doc.page_end + 1}",
                'chunks': len(doc.chunks) if doc.chunks else 0,
                'preview': doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
            }
            for doc in self.logical_docs
        ]

    def _get_search_type_label(self) -> str:
        label = 'Hybrid (Pinecone + BM25 with RRF)'
        if self.use_rerank:
            label += ' + Rerank'
        return label
