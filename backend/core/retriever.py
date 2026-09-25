"""retriever.py — Hybrid retrieval system using Pinecone sparse-dense index."""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

from core.models import ChunkMetadata, SearchConfig
from llm.llm_router import EMBED_DIM


class HybridRetriever:
    """    Hybrid retrieval system using Pinecone's sparse-dense index."""

    def __init__(self, namespace: str, chat_id: str, alpha: float = 0.5):
        """        Initialize the Pinecone hybrid retriever."""
        self.alpha = alpha

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        pinecone_host = os.getenv("PINECONE_HOST")
        self.namespace = namespace
        self.chat_id = chat_id
        self.id_prefix = f"{chat_id}#"

        if not all([pinecone_api_key, pinecone_index_name, pinecone_host]):
            print("⚠️ PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_HOST must be set in .env for Pinecone retrieval.")
            self.pc_index = None
        else:
            pc = Pinecone(api_key=pinecone_api_key)
            self.pc_index = pc.Index(pinecone_index_name, host=pinecone_host)
            self._assert_index_compatible(pc, pinecone_index_name)
            print(f"✅ Pinecone index '{pinecone_index_name}' [namespace: {self.namespace}] configured successfully.")

        self.bm25_encoder = BM25Encoder()

        self.chunks_metadata: List[ChunkMetadata] = []

        self.embed_model = None

        self.chunk_count: int = 0

    def _assert_index_compatible(self, pc, index_name: str):
        """        Fail at session start, not after a full extraction run."""
        try:
            spec = pc.describe_index(index_name)
        except Exception as e:
            print(f"⚠️ Could not describe index '{index_name}': {type(e).__name__}: {e}")
            return

        if spec.dimension != EMBED_DIM:
            raise ValueError(
                f"Pinecone index '{index_name}' has dimension {spec.dimension}, "
                f"but embeddings are {EMBED_DIM}-dimensional. Recreate the index "
                f"with dimension={EMBED_DIM} and metric='dotproduct'."
            )
        if spec.metric != "dotproduct":
            raise ValueError(
                f"Pinecone index '{index_name}' uses metric '{spec.metric}'. "
                f"Sparse-dense hybrid search requires metric='dotproduct'."
            )

    def _list_chat_ids(self) -> List[str]:
        """        Every vector id belonging to this chat."""
        ids: List[str] = []
        for page in self.pc_index.list(prefix=self.id_prefix,
                                       namespace=self.namespace):
            for item in getattr(page, "vectors", page) or []:
                ids.append(getattr(item, "id", item))
        return ids

    def delete_chat(self):
        """        Delete only this chat's vectors, leaving the rest of the namespace."""
        if self.pc_index is None:
            return
        try:
            ids = self._list_chat_ids()
            for i in range(0, len(ids), 1000):
                self.pc_index.delete(ids=ids[i:i + 1000], namespace=self.namespace)
            print(f"🧹 Deleted {len(ids)} vectors for chat '{self.chat_id}'.")
        except Exception as e:
            print(f"ℹ️ Could not delete chat '{self.chat_id}': "
                  f"{type(e).__name__}: {e}")

    def delete_namespace(self):
        """Delete every chat in this namespace. Used when an account goes."""
        if self.pc_index is None:
            return
        try:
            self.pc_index.delete_namespace(name=self.namespace)
            print(f"🧹 Deleted Pinecone namespace '{self.namespace}'.")
        except Exception as e:
            print(f"ℹ️ Could not delete namespace '{self.namespace}': "
                  f"{type(e).__name__}: {e}")

    def all_chunks(self) -> List[ChunkMetadata]:
        """Every chunk of this chat's document, in reading order."""
        if self.pc_index is None:
            return []
        ids = self._list_chat_ids()
        chunks: List[ChunkMetadata] = []
        for i in range(0, len(ids), 100):
            resp = self.pc_index.fetch(ids=ids[i:i + 100], namespace=self.namespace)
            vectors = getattr(resp, "vectors", None)
            if vectors is None and isinstance(resp, dict):
                vectors = resp.get("vectors")
            for vid, vec in (vectors or {}).items():
                md = getattr(vec, "metadata", None)
                if md is None and isinstance(vec, dict):
                    md = vec.get("metadata")
                chunks.append(self._match_to_chunk({"id": vid, "metadata": md or {}}))
        chunks.sort(key=lambda c: (c.page_start, c.chunk_index))
        return chunks


    def build_indices(self, chunks_metadata: List[ChunkMetadata], embed_model, on_stage=None):
        """        Build Pinecone hybrid index with dense + sparse vectors.
        Upserts into existing Pinecone index."""
        if not self.pc_index:
            raise ValueError("Pinecone index not configured. Check .env variables.")

        print("🔨 Building Pinecone hybrid index...")

        self.chunks_metadata = chunks_metadata
        self.embed_model = embed_model

        if on_stage:
            on_stage("embed")
        print("  📊 Computing dense embeddings...")

        def _augmented(c):
            """Contextual Retrieval: prepend the per-document identity so BM25 and
            dense embedding both see 'James Bond' next to his value cell. Kept
            out of `chunk.text` so citation previews stay clean."""
            return f"[{c.context}]\n{c.text}" if c.context else c.text

        texts = [_augmented(chunk) for chunk in chunks_metadata]
        dense_embeddings = embed_model.encode(
            texts, show_progress_bar=True, task_type="RETRIEVAL_DOCUMENT"
        )
        dense_embeddings = np.array(dense_embeddings).astype('float32')

        doc_types = set(chunk.doc_type for chunk in chunks_metadata)

        print("  📝 Fitting BM25 encoder on corpus...")
        self.bm25_encoder.fit(texts)
        print("  ✅ BM25 encoder fitted")

        print("  📝 Generating sparse vectors...")
        sparse_vectors = [self.bm25_encoder.encode_documents(text) for text in texts]

        if on_stage:
            on_stage("store")
        print(f"  📤 Upserting vectors into Pinecone [namespace: {self.namespace}]...")
        batch_size = 100

        for i in range(0, len(chunks_metadata), batch_size):
            batch_end = min(i + batch_size, len(chunks_metadata))
            vectors_to_upsert = []

            for j in range(i, batch_end):
                chunk = chunks_metadata[j]

                dense_vec = dense_embeddings[j].copy()
                norm = np.linalg.norm(dense_vec)
                if norm > 0:
                    dense_vec = dense_vec / norm

                vectors_to_upsert.append({
                    "id": f"{self.id_prefix}{chunk.chunk_id}",
                    "values": dense_vec.tolist(),
                    "sparse_values": sparse_vectors[j],
                    "metadata": {
                        "chat_id": self.chat_id,
                        "text": chunk.text,
                        "context": chunk.context or "",
                        "doc_type": chunk.doc_type,
                        "doc_id": chunk.doc_id,
                        "filename": chunk.filename,
                        "chunk_index": chunk.chunk_index,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end
                    }
                })

            self.pc_index.upsert(vectors=vectors_to_upsert, namespace=self.namespace)

            if (i // batch_size) % 5 == 0:
                print(f"    Upserted {batch_end}/{len(chunks_metadata)} vectors...")

        self.chunk_count = len(chunks_metadata)

        print(f"✅ Indexed {len(chunks_metadata)} chunks across {len(doc_types)} document types")
        print(f"   Alpha={self.alpha}")


    def export_bm25_params(self) -> Dict:
        """The fitted BM25 encoder, as a JSON-serialisable dict."""
        return self.bm25_encoder.get_params()

    def rehydrate(self, bm25_params: Dict, embed_model,
                  chunk_count: int = 0) -> None:
        """        Restore a retriever for a chat indexed in an earlier process, without
        re-ingesting the document."""
        self.embed_model = embed_model
        self.bm25_encoder.set_params(**bm25_params)
        self._load_corpus_shape_from_index(chunk_count)

    def _load_corpus_shape_from_index(self, chunk_count: int = 0) -> None:
        """        Set this chat's vector count."""
        if chunk_count:
            self.chunk_count = chunk_count
            print(f"♻️ Rehydrated chat '{self.chat_id}': {chunk_count} chunks")
            return
        if self.pc_index is None:
            return
        try:
            self.chunk_count = len(self._list_chat_ids())
            if self.chunk_count:
                print(f"♻️ Rehydrated chat '{self.chat_id}': {self.chunk_count} chunks")
            else:
                print(f"⚠️ Chat '{self.chat_id}' has no vectors in '{self.namespace}'.")
        except Exception as e:
            print(f"ℹ️ Could not count vectors for '{self.chat_id}': "
                  f"{type(e).__name__}: {e}")


    def _resolve_search_config(self, filter_doc_type: Optional[str]) -> SearchConfig:
        """        Decide the search scope. Only the caller can narrow it."""
        if filter_doc_type:
            print(f"🔍 Using explicit filter: {filter_doc_type}")
            return SearchConfig(
                use_filtered=True,
                selected_type=filter_doc_type,
                total_chunks=self.chunk_count,
                routing_info={'method': 'user_filter', 'type': filter_doc_type},
            )

        return SearchConfig(
            use_filtered=False,
            selected_type=None,
            total_chunks=self.chunk_count,
            routing_info={'method': 'full_search'},
        )


    def _scale_vectors(self, dense_vec: List[float], sparse_vec: Dict) -> Tuple[List[float], Dict]:
        """        Scale dense and sparse vectors using alpha for hybrid search."""
        scaled_dense = [v * self.alpha for v in dense_vec]

        scaled_sparse = {
            "indices": sparse_vec["indices"],
            "values": [v * (1 - self.alpha) for v in sparse_vec["values"]]
        }

        return scaled_dense, scaled_sparse

    def _query_pinecone(self, query: str, top_k: int,
                         filter_doc_type: Optional[str] = None) -> List[Dict]:
        """
        Query Pinecone hybrid index with both dense and sparse vectors.
        Single API call — Pinecone searches dense + sparse and fuses internally.
        """
        query_dense = np.asarray(
            self.embed_model.encode([query], task_type="RETRIEVAL_QUERY")[0],
            dtype='float32',
        )
        norm = np.linalg.norm(query_dense)
        if norm > 0:
            query_dense = query_dense / norm

        query_sparse = self.bm25_encoder.encode_queries(query)

        scaled_dense, scaled_sparse = self._scale_vectors(
            query_dense.tolist(), query_sparse
        )

        query_filter = {"chat_id": {"$eq": self.chat_id}}
        if filter_doc_type:
            query_filter["doc_type"] = {"$eq": filter_doc_type}

        query_params = {
            "vector": scaled_dense,
            "sparse_vector": scaled_sparse,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": self.namespace,
            "filter": query_filter,
        }

        results = self.pc_index.query(**query_params)

        return results.get("matches", [])

    @staticmethod
    def _match_to_chunk(match: Dict) -> ChunkMetadata:
        """        Rebuild a ChunkMetadata from a Pinecone match."""
        md = match.get("metadata") or {}
        return ChunkMetadata(
            chunk_id=match["id"],
            doc_id=md.get("doc_id", "unknown"),
            doc_type=md.get("doc_type", "Other"),
            filename=md.get("filename", "unknown"),
            chunk_index=int(md.get("chunk_index", 0)),
            page_start=int(md.get("page_start", 0)),
            page_end=int(md.get("page_end", 0)),
            text=md.get("text", ""),
            context=(md.get("context") or None),
        )


    def retrieve(self, query: str, k: int = 6,
                 filter_doc_type: Optional[str] = None,
                 return_details: bool = False) -> Union[List[Tuple], Dict]:
        """        Retrieve relevant chunks using Pinecone hybrid search."""
        if self.pc_index is None:
            raise ValueError("No index connected. Check your Pinecone .env variables.")

        config = self._resolve_search_config(filter_doc_type)
        scope = config.selected_type if config.use_filtered else "all chunks"
        print(f"📊 Searching {scope} of {config.total_chunks} (alpha={self.alpha})")

        matches = self._query_pinecone(
            query=query,
            top_k=k,
            filter_doc_type=config.selected_type if config.use_filtered else None
        )

        results = [(self._match_to_chunk(m), m["score"]) for m in matches]

        if not results:
            print("⚠️ No results found")
            if return_details:
                return {'results': [], 'routing_info': config.routing_info, 'query': query}
            return []

        results = results[:k]

        if not return_details:
            return results

        detailed_results = []
        for chunk, final_score in results:
            detailed_results.append({
                'chunk': chunk,
                'chunk_index': chunk.chunk_index,
                'final_score': final_score,
                'doc_type': chunk.doc_type,
                'pages': f"{chunk.page_start + 1}-{chunk.page_end + 1}",
            })

        return {
            'results': detailed_results,
            'routing_info': config.routing_info,
            'config': {
                'alpha': self.alpha,
            },
            'retrieval_stats': {
                'total_chunks': config.total_chunks,
                'final_returned': len(detailed_results)
            },
            'query': query
        }
