"""
document_store.py — The main orchestration layer.

Glues together PDF processing, chunking, retrieval, and generation into
a single stateful `EnhancedDocumentStoreHybrid` class that can be cleanly
used by the UI.
"""

from datetime import datetime
from typing import List, Dict, Optional, Tuple

from core.pdf_processor import extract_and_analyze_pdf
from core.chunker import process_all_documents
from core.retriever import HybridRetriever
from core.answer_generator import generate_answer_with_sources


class EnhancedDocumentStoreHybrid:
    """
    Manages the complete document processing and retrieval pipeline.
    Uses Pinecone sparse-dense hybrid search with alpha-based fusion
    with alpha-based fusion.
    """

    def __init__(self, namespace: str, chat_id: str, alpha: float = 0.5):
        """
        Args:
            namespace: the owning user's Pinecone namespace, shared by all
                       their chats.
            chat_id:   which chat inside it. Scopes every query and delete.
        """
        self.namespace = namespace
        self.chat_id = chat_id
        self.pages_info = []
        self.logical_docs = []
        self.chunks_metadata = []

        self.retriever = HybridRetriever(namespace=namespace, chat_id=chat_id,
                                         alpha=alpha)
        self.alpha = alpha

        self.is_ready = False
        self.processing_stats = {}
        self.filename = None

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    @classmethod
    def rehydrate(cls, namespace: str, chat_id: str, bm25_params: Dict,
                  doc_stats: Dict, embed_model,
                  alpha: float = 0.5) -> "EnhancedDocumentStoreHybrid":
        """
        Rebuild a store for a document indexed in an earlier process.

        Skips the entire ingestion pipeline — no Docling, no classification, no
        re-embedding. The vectors are already in Pinecone; only the fitted BM25
        encoder (from Postgres) has to be restored.
        """
        store = cls(namespace=namespace, chat_id=chat_id, alpha=alpha)
        store.retriever.rehydrate(bm25_params, embed_model,
                                  chunk_count=(doc_stats or {}).get("total_chunks", 0))
        store.processing_stats = doc_stats or {}
        store.filename = store.processing_stats.get("filename")
        store.is_ready = True
        return store

    def export_bm25_params(self) -> Dict:
        """Fitted BM25 encoder, for persisting alongside the chat session."""
        return self.retriever.export_bm25_params()

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
                'structure': self.get_document_structure(),
            }

            self.is_ready = True
            return True, self.processing_stats

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, {'error': str(e)}

    def query(
        self, question: str, filter_type: Optional[str] = None,
        k: int = 6, return_details: bool = False
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
            return_details=return_details
        )

        # retrieve() returns a details dict or a bare list of (chunk, score)
        # depending on return_details.
        if return_details:
            retrieved = [(r['chunk'], r['final_score'])
                         for r in retrieval_result.get('results', [])]
        else:
            retrieved = list(retrieval_result)

        result = generate_answer_with_sources(question, retrieved)
        if return_details:
            result['retrieval_details'] = retrieval_result

        result['filter_used'] = filter_type or 'none'
        return result

    def retrieve_only(
        self, question: str, filter_type: Optional[str] = None, k: int = 6
    ) -> List[Tuple]:
        """Retrieve without generating — the streaming path generates its own
        tokens, so it needs the chunks but not a buffered answer. Returns the
        same (chunk, score) list query() feeds to the generator."""
        if not self.is_ready:
            return []
        retrieval_result = self.retriever.retrieve(
            question, k=k, filter_doc_type=filter_type, return_details=True
        )
        return [(r['chunk'], r['final_score'])
                for r in retrieval_result.get('results', [])]

    def get_document_structure(self) -> List[Dict]:
        """Summarise the ingested documents for the UI dashboard."""
        if not self.logical_docs:
            # Rehydrated store: logical_docs was never rebuilt, but the structure
            # was snapshotted into doc_stats at ingest time.
            return self.processing_stats.get('structure', [])
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

    def set_alpha(self, alpha: float):
        """Update the dense/sparse balance for hybrid search."""
        self.alpha = alpha
        self.retriever.alpha = alpha
        print(f"🔄 Alpha updated to {alpha}")

    def _get_search_type_label(self) -> str:
        if self.alpha == 1.0:
            label = 'Vector (pure semantic)'
        elif self.alpha == 0.0:
            label = 'BM25 (pure keyword)'
        else:
            label = f'Hybrid (alpha={self.alpha})'
        return label
