"""
retriever.py — Hybrid retrieval system combining FAISS, BM25, and RRF.

Provides the HybridRetriever class, which handles:
  - Building chunk indices (vector & keyword)
  - Embedding centroid classification for auto-routing (no LLM needed)
  - Reciprocal Rank Fusion (RRF) for scoring
  - Fallback to full-search when filtered results are weak
  - Optional CrossEncoder reranking
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict, Union
from rank_bm25 import BM25Okapi

from core.models import ChunkMetadata, SearchConfig


class HybridRetriever:
    """
    Hybrid retrieval system using Reciprocal Rank Fusion (RRF).

    Combines:
    1. FAISS vector search (semantic similarity)
    2. BM25 keyword search (lexical matching)
    3. Embedding centroid classification for document-type routing (no LLM)
    4. Optional CrossEncoder reranking for improved relevance

    Uses equal-weight RRF for robust score combination.
    Centroid routing classifies the query to a document type by comparing
    its embedding against per-type centroid vectors.
    """

    def __init__(self,
                 rrf_k: int = 60,
                 score_floor: float = 0.05,
                 use_rerank: bool = False,
                 margin_threshold: float = 0.09,
                 fallback_score_threshold: float = 0.15):
        """
        Initialize the hybrid retriever.

        Args:
            rrf_k: RRF constant (default 60, as in the original paper)
            score_floor: Minimum normalized score (default 0.05)
            use_rerank: Whether to use reranking (default False)
            margin_threshold: Minimum margin between top two centroids
                              to trust the classification (default 0.09)
            fallback_score_threshold: If filtered search top score is below
                                      this, re-search all chunks (default 0.15)
        """
        self.rrf_k = rrf_k
        self.score_floor = score_floor

        # Centroid classification parameters
        self.margin_threshold = margin_threshold
        self.fallback_score_threshold = fallback_score_threshold

        # Reranking configuration
        self.use_rerank = use_rerank
        self.reranker = None
        if self.use_rerank:
            self._load_reranker()

        # Main indices
        self.faiss_index = None
        self.bm25_index = None
        self.chunks_metadata: List[ChunkMetadata] = []
        self.tokenized_corpus: List[List[str]] = []

        # Document type specific indices
        self.doc_type_indices: Dict[str, Dict] = {}

        # Centroid vectors for document-type classification
        self.doc_type_centroids: Dict[str, np.ndarray] = {}

        # Embedding model reference (set during build)
        self.embed_model = None

    def clear(self):
        """Reset all indices and loaded data."""
        self.faiss_index = None
        self.bm25_index = None
        self.chunks_metadata = []
        self.tokenized_corpus = []
        self.doc_type_indices = {}
        self.doc_type_centroids = {}
        self.embed_model = None

    # -----------------------------------------------------------------------
    # Reranker Initialisation
    # -----------------------------------------------------------------------

    def _load_reranker(self):
        """
        Set up the reranker.

        Checks for RERANKER_URL env var first:
          - If set → use Modal HTTP endpoint (no local model loaded)
          - Otherwise → load BAAI/bge-reranker-v2-m3 locally
        """
        import os
        modal_url = os.getenv("RERANKER_URL", "").strip()

        if modal_url:
            # ---- Modal remote reranker ----
            print(f"🌐 Using Modal reranker endpoint: {modal_url}")
            self.reranker = modal_url   # store URL string as the 'reranker'
            self._rerank_mode = "modal"
        else:
            # ---- Local CrossEncoder fallback ----
            try:
                from sentence_transformers import CrossEncoder
                import torch
                print("🔄 Loading BAAI/bge-reranker-v2-m3 locally...")
                self.reranker = CrossEncoder(
                    "BAAI/bge-reranker-v2-m3",
                    max_length=1024,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                self._rerank_mode = "local"
                print("✅ Reranker model loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load reranker: {e}")
                print("   Continuing without reranking...")
                self.use_rerank = False
                self.reranker = None
                self._rerank_mode = None

    def set_rerank(self, use_rerank: bool):
        """Enable or disable reranking."""
        if use_rerank and not self.reranker:
            self.use_rerank = True
            self._load_reranker()
        elif not use_rerank:
            self.use_rerank = False
            print("🔄 Reranking disabled")
        else:
            self.use_rerank = True
            mode = getattr(self, '_rerank_mode', 'local')
            print(f"✅ Reranking enabled (mode: {mode})")

    # -----------------------------------------------------------------------
    # Index Building
    # -----------------------------------------------------------------------

    def build_indices(self, chunks_metadata: List[ChunkMetadata], embed_model):
        """Build FAISS, BM25, and centroid indices."""
        rerank_status = "enabled" if self.use_rerank else "disabled"
        print(f"🔨 Building hybrid indices (FAISS + BM25 + Centroids, rerank: {rerank_status})...")

        self.chunks_metadata = chunks_metadata
        self.embed_model = embed_model

        # === Compute Embeddings ===
        print("  📊 Computing embeddings...")
        texts = [chunk.text for chunk in chunks_metadata]
        embeddings = embed_model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # Store RAW (unnormalized) embeddings in metadata — used for centroids
        for i, chunk in enumerate(chunks_metadata):
            chunk.embedding = embeddings[i].copy()

        # === Build Centroid Vectors (before normalization) ===
        print("  🎯 Computing document-type centroids...")
        doc_types = set(chunk.doc_type for chunk in chunks_metadata)

        for doc_type in doc_types:
            type_indices = [i for i, chunk in enumerate(chunks_metadata)
                            if chunk.doc_type == doc_type]
            if type_indices:
                type_embeddings = embeddings[type_indices]
                centroid = type_embeddings.mean(axis=0)
                # Normalize centroid for cosine similarity
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                self.doc_type_centroids[doc_type] = centroid

        print(f"  ✅ Built {len(self.doc_type_centroids)} document-type centroids")

        # === FAISS Vector Index (normalize a COPY — do not mutate raw embeddings) ===
        print("  📊 Building FAISS vector index...")
        faiss_embeddings = embeddings.copy()
        dim = faiss_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(faiss_embeddings)
        self.faiss_index.add(faiss_embeddings)

        # === BM25 Keyword Index ===
        print("  📝 Building BM25 keyword index...")
        self.tokenized_corpus = [self._bm25_tokenize(chunk.text) for chunk in chunks_metadata]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)

        # === Document Type Specific Indices ===
        print("  🏷️ Building document type specific indices...")

        for doc_type in doc_types:
            type_indices = [i for i, chunk in enumerate(chunks_metadata)
                            if chunk.doc_type == doc_type]
            if type_indices:
                # FAISS index for this type (use the already-normalized faiss_embeddings)
                type_faiss_embeddings = faiss_embeddings[type_indices]
                type_faiss_index = faiss.IndexFlatIP(dim)
                type_faiss_index.add(type_faiss_embeddings)

                # BM25 index for this type
                type_tokenized = [self.tokenized_corpus[i] for i in type_indices]
                type_bm25_index = BM25Okapi(type_tokenized)

                self.doc_type_indices[doc_type] = {
                    'faiss_index': type_faiss_index,
                    'bm25_index': type_bm25_index,
                    'mapping': type_indices,
                    'tokenized': type_tokenized
                }

        print(f"✅ Indexed {len(chunks_metadata)} chunks across {len(doc_types)} document types")
        print(f"   RRF k={self.rrf_k}, margin_threshold={self.margin_threshold}")

    # -----------------------------------------------------------------------
    # Tokenization helper (local — no longer needs query_router import)
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_text(s: str) -> str:
        import re
        s = s.lower()
        s = s.replace(",", "")
        s = s.replace("$", "")
        s = s.replace("\u00a0", " ")
        import re
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _bm25_tokenize(s: str) -> List[str]:
        import re
        s = HybridRetriever._normalize_text(s)
        return re.findall(r"[a-z]+|\d+(?:\.\d+)?%?", s)

    # -----------------------------------------------------------------------
    # Centroid Classification (replaces LLM-based routing)
    # -----------------------------------------------------------------------

    def _classify_by_centroid(self, query: str) -> Tuple[Optional[str], float, Dict]:
        """
        Classify query to a document type using embedding centroid similarity.

        Embeds the query, compares against all document-type centroids,
        and uses the margin between top two scores as confidence.

        Args:
            query: User's search query

        Returns:
            Tuple of (predicted_type, margin, routing_info_dict)
            predicted_type is None if no centroids exist
        """
        if not self.doc_type_centroids:
            return None, 0.0, {'method': 'no_centroids'}

        # Embed and normalize the query
        query_embedding = self.embed_model.encode([query])[0].astype('float32')
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # Compute cosine similarity against all centroids
        scores = {}
        for doc_type, centroid in self.doc_type_centroids.items():
            scores[doc_type] = float(np.dot(query_embedding, centroid))

        # Sort by score descending
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        best_type = sorted_types[0][0]
        best_score = sorted_types[0][1]

        # Compute margin between top two
        if len(sorted_types) >= 2:
            second_score = sorted_types[1][1]
            margin = best_score - second_score
        else:
            margin = best_score  # Only one type exists

        routing_info = {
            'method': 'centroid',
            'predicted_type': best_type,
            'best_score': round(best_score, 4),
            'margin': round(margin, 4),
            'top_3': [(t, round(s, 4)) for t, s in sorted_types[:3]]
        }

        print(f"🎯 Centroid classification: {best_type} "
              f"(score: {best_score:.4f}, margin: {margin:.4f})")

        return best_type, margin, routing_info

    # -----------------------------------------------------------------------
    # Core Retrieval & Scoring Logic
    # -----------------------------------------------------------------------

    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Apply min-max normalisation with a score floor."""
        if not scores:
            return {}
        min_score = min(scores.values())
        max_score = max(scores.values())

        if max_score > min_score:
            return {
                idx: self.score_floor + (1.0 - self.score_floor) *
                     (score - min_score) / (max_score - min_score)
                for idx, score in scores.items()
            }
        return {idx: 1.0 for idx in scores}

    def _resolve_search_config(
        self, query: str, filter_doc_type: Optional[str],
        auto_route: bool, confidence_threshold: float
    ) -> SearchConfig:
        """Decide search scope: explicit filter → centroid routing → full search."""
        use_filtered = False
        selected_type = None
        routing_info = {'method': 'full_search'}

        # Priority 1: Explicit filter from UI
        if filter_doc_type and filter_doc_type in self.doc_type_indices:
            use_filtered = True
            selected_type = filter_doc_type
            routing_info = {'method': 'filter', 'type': filter_doc_type}
            print(f"🔍 Using explicit filter: {filter_doc_type}")

        # Priority 2: Auto-route using centroid classification (no LLM)
        elif auto_route and self.doc_type_centroids:
            predicted_type, margin, centroid_info = self._classify_by_centroid(query)
            routing_info = centroid_info

            if margin >= self.margin_threshold and predicted_type in self.doc_type_indices:
                use_filtered = True
                selected_type = predicted_type
                print(f"✅ Centroid routing to: {predicted_type} (margin: {margin:.4f})")
            else:
                print(f"⚠️ Low margin ({margin:.4f} < {self.margin_threshold}), "
                      f"searching all chunks")

        total_chunks = (len(self.doc_type_indices[selected_type]['mapping'])
                        if use_filtered else len(self.chunks_metadata))

        return SearchConfig(
            use_filtered=use_filtered,
            selected_type=selected_type,
            total_chunks=total_chunks,
            routing_info=routing_info
        )

    def _get_faiss_scores(self, query: str, k: int, faiss_index,
                          chunk_indices: List[int] = None) -> Dict[int, float]:
        """Get raw FAISS dot-product scores."""
        query_embedding = self.embed_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)

        actual_k = min(k, faiss_index.ntotal)
        scores, indices = faiss_index.search(query_embedding, actual_k)

        result = {}
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            original_idx = chunk_indices[idx] if chunk_indices else idx
            result[original_idx] = float(score)
        return result

    def _get_bm25_scores(self, query: str, k: int, bm25_index,
                         chunk_indices: List[int] = None) -> Dict[int, float]:
        """Get raw BM25 token-match scores."""
        query_tokens = self._bm25_tokenize(query)
        scores = bm25_index.get_scores(query_tokens)

        actual_k = min(k, len(scores))
        top_indices = np.argsort(scores)[::-1][:actual_k]

        result = {}
        for idx in top_indices:
            if scores[idx] > 0:
                original_idx = chunk_indices[idx] if chunk_indices else idx
                result[original_idx] = float(scores[idx])
        return result

    def _get_all_scores(self, query: str, config: SearchConfig
                        ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Run both FAISS and BM25 retrievals."""
        if config.use_filtered:
            tdata = self.doc_type_indices[config.selected_type]
            faiss_s = self._get_faiss_scores(query, config.total_chunks,
                                             tdata['faiss_index'], tdata['mapping'])
            bm25_s = self._get_bm25_scores(query, config.total_chunks,
                                            tdata['bm25_index'], tdata['mapping'])
        else:
            faiss_s = self._get_faiss_scores(query, config.total_chunks, self.faiss_index)
            bm25_s = self._get_bm25_scores(query, config.total_chunks, self.bm25_index)
        return faiss_s, bm25_s

    def _combine_scores_rrf(self, faiss_scores: Dict[int, float],
                            bm25_scores: Dict[int, float]) -> Dict[int, float]:
        """Combine lists using Reciprocal Rank Fusion (equal 50/50 weighting)."""
        faiss_ranking = sorted(faiss_scores.keys(), key=lambda x: faiss_scores[x], reverse=True)
        bm25_ranking = sorted(bm25_scores.keys(), key=lambda x: bm25_scores[x], reverse=True)

        faiss_ranks = {idx: rank + 1 for rank, idx in enumerate(faiss_ranking)}
        bm25_ranks = {idx: rank + 1 for rank, idx in enumerate(bm25_ranking)}

        all_indices = set(faiss_scores.keys()) | set(bm25_scores.keys())
        combined = {}

        for idx in all_indices:
            rrf_score = 0.0
            if idx in faiss_ranks:
                rrf_score += 0.5 * (1.0 / (self.rrf_k + faiss_ranks[idx]))
            if idx in bm25_ranks:
                rrf_score += 0.5 * (1.0 / (self.rrf_k + bm25_ranks[idx]))
            combined[idx] = rrf_score

        return self._normalize_scores(combined)

    def _rerank_results(self, query: str, results: List[Tuple],
                        top_k: int = None) -> List[Tuple]:
        """
        Rerank candidates using a strong CrossEncoder.

        Supports two modes:
          - 'modal' : POST scores to the Modal HTTP endpoint (no local GPU needed)
          - 'local' : run CrossEncoder directly on local device

        Args:
            results: List of (chunk_idx, ChunkMetadata, score) tuples
        """
        if not self.use_rerank or not self.reranker or not results:
            return results

        texts = [chunk.text for _, chunk, _ in results]
        print(f"🔄 Reranking {len(results)} results (mode: {getattr(self, '_rerank_mode', 'local')})...")

        # ---- Get raw scores ----
        try:
            if getattr(self, '_rerank_mode', 'local') == "modal":
                import requests
                resp = requests.post(
                    f"{self.reranker}/rerank",
                    json={"query": query, "texts": texts},
                    timeout=60,
                )
                resp.raise_for_status()
                rerank_scores = resp.json()["scores"]
            else:
                rerank_scores = self.reranker.predict([[query, t] for t in texts])
        except Exception as e:
            print(f"⚠️ Reranking failed ({e}), returning original order")
            return results[:top_k] if top_k else results

        # ---- Normalise to [score_floor, 1.0] ----
        min_score, max_score = min(rerank_scores), max(rerank_scores)
        if max_score > min_score:
            normalized_scores = [
                self.score_floor + (1.0 - self.score_floor) * (s - min_score) / (max_score - min_score)
                for s in rerank_scores
            ]
        else:
            normalized_scores = [1.0] * len(rerank_scores)

        reranked = [(idx, chunk, float(norm))
                    for (idx, chunk, _), norm in zip(results, normalized_scores)]
        reranked.sort(key=lambda x: x[2], reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        print(f"✅ Reranking complete. Top score: {reranked[0][2]:.4f}")
        return reranked

    # -----------------------------------------------------------------------
    # Main Retriever Interface
    # -----------------------------------------------------------------------

    def retrieve(
        self, query: str, k: int = 4, filter_doc_type: Optional[str] = None,
        auto_route: bool = False, return_details: bool = False,
        confidence_threshold: float = 0.7, search_mode: str = "hybrid"
    ) -> Union[List[Tuple], Dict]:
        """Execute full retrieval pipeline and return top-k chunks."""
        if self.faiss_index is None and self.bm25_index is None:
            raise ValueError("No indices built. Call build_indices() first.")

        # === Step 1: Resolve search config via centroid routing ===
        config = self._resolve_search_config(
            query, filter_doc_type, auto_route, confidence_threshold
        )
        print(f"📊 Searching {config.total_chunks} chunks with mode: {search_mode.upper()}")

        # === Step 2: Get scores based on search mode ===
        if search_mode == "vector":
            if config.use_filtered:
                type_data = self.doc_type_indices[config.selected_type]
                faiss_scores = self._get_faiss_scores(
                    query, config.total_chunks,
                    type_data['faiss_index'], type_data['mapping']
                )
            else:
                faiss_scores = self._get_faiss_scores(
                    query, config.total_chunks, self.faiss_index
                )
            combined_scores = self._normalize_scores(faiss_scores)
            bm25_scores = {}

        elif search_mode == "bm25":
            if config.use_filtered:
                type_data = self.doc_type_indices[config.selected_type]
                bm25_scores = self._get_bm25_scores(
                    query, config.total_chunks,
                    type_data['bm25_index'], type_data['mapping']
                )
            else:
                bm25_scores = self._get_bm25_scores(
                    query, config.total_chunks, self.bm25_index
                )
            combined_scores = self._normalize_scores(bm25_scores)
            faiss_scores = {}

        else:  # Default: hybrid
            faiss_scores, bm25_scores = self._get_all_scores(query, config)
            combined_scores = self._combine_scores_rrf(faiss_scores, bm25_scores)

        # === Step 3: Fallback — if filtered top score is too weak ===
        if combined_scores and config.use_filtered:
            top_score = max(combined_scores.values())
            if top_score < self.fallback_score_threshold:
                print(f"⚠️ Filtered top score ({top_score:.4f}) below threshold "
                      f"({self.fallback_score_threshold}), falling back to full search")

                config = SearchConfig(
                    use_filtered=False,
                    selected_type=None,
                    total_chunks=len(self.chunks_metadata),
                    routing_info={
                        'method': 'fallback',
                        'original': config.routing_info,
                        'reason': f'top_score {top_score:.4f} < {self.fallback_score_threshold}'
                    }
                )

                if search_mode == "vector":
                    faiss_scores = self._get_faiss_scores(
                        query, config.total_chunks, self.faiss_index
                    )
                    combined_scores = self._normalize_scores(faiss_scores)
                    bm25_scores = {}
                elif search_mode == "bm25":
                    bm25_scores = self._get_bm25_scores(
                        query, config.total_chunks, self.bm25_index
                    )
                    combined_scores = self._normalize_scores(bm25_scores)
                    faiss_scores = {}
                else:
                    faiss_scores, bm25_scores = self._get_all_scores(query, config)
                    combined_scores = self._combine_scores_rrf(faiss_scores, bm25_scores)

        # === Step 4: Sort and build results carrying chunk_index ===
        if not combined_scores:
            print(f"⚠️ No scores returned for mode: {search_mode}")
            if return_details:
                return {'results': [], 'routing_info': config.routing_info, 'query': query}
            return []

        sorted_indices = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True
        )

        # Internal tuple: (chunk_idx, ChunkMetadata, score)
        results = [
            (idx, self.chunks_metadata[idx], combined_scores[idx])
            for idx in sorted_indices
        ]

        # === Step 5: Reranking ===
        rerank_applied = False
        if self.use_rerank and self.reranker and len(results) > 1:
            results = self._rerank_results(query, results, top_k=k)
            rerank_applied = True
        else:
            results = results[:k]

        # === Step 6: Return ===
        if not return_details:
            # Strip index — return List[(ChunkMetadata, score)]
            return [(chunk, score) for _, chunk, score in results]

        detailed_results = []
        for chunk_idx, chunk, final_score in results:
            detailed_results.append({
                'chunk': chunk,
                'chunk_index': chunk_idx,
                'final_score': final_score,
                'faiss_score': faiss_scores.get(chunk_idx, 0.0),
                'bm25_score': bm25_scores.get(chunk_idx, 0.0),
                'rrf_score': combined_scores.get(chunk_idx, 0.0) if search_mode == "hybrid" else final_score,
                'in_faiss_top': chunk_idx in faiss_scores,
                'in_bm25_top': chunk_idx in bm25_scores,
                'doc_type': chunk.doc_type,
                'pages': f"{chunk.page_start}-{chunk.page_end}",
                'reranked': rerank_applied
            })

        return {
            'results': detailed_results,
            'routing_info': config.routing_info,
            'search_mode': search_mode,
            'rrf_config': {
                'k': self.rrf_k if search_mode == "hybrid" else None,
                'score_floor': self.score_floor,
                'rerank_enabled': self.use_rerank,
                'rerank_applied': rerank_applied
            },
            'retrieval_stats': {
                'total_chunks': config.total_chunks,
                'final_returned': len(detailed_results)
            },
            'query': query
        }
