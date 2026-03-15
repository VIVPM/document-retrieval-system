"""
main.py — FastAPI backend for the Document Retrieval System.

Endpoints:
  POST /upload          — Upload and process a PDF file
  POST /query           — Ask a question about the processed document
  GET  /structure       — Get the document structure (types + pages)
  GET  /status          — Check if a document has been processed
  POST /clear           — Clear all stored documents and reset state
  POST /settings/rerank — Enable or disable the BGE reranker
"""

import os
import tempfile
import shutil

import multiprocess.resource_tracker as rt

# Suppress harmless Windows multiprocess exit error
def _silent_del(self):
    try:
        self._stop()
    except Exception:
        pass

rt.ResourceTracker.__del__ = _silent_del

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from llm.llm_router import embed_model
from core.document_store import EnhancedDocumentStoreHybrid


def sanitize(obj):
    """
    Recursively convert non-JSON-serializable types to native Python.
    Handles: numpy scalars/arrays, dataclasses, objects with __dict__,
    sets, bytes, and arbitrary objects.
    """
    import numpy as np
    from dataclasses import asdict, is_dataclass

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, set):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if is_dataclass(obj) and not isinstance(obj, type):
        return sanitize(asdict(obj))
    if hasattr(obj, '__dict__'):
        return sanitize(vars(obj))
    # Last resort — stringify
    return str(obj)



# ── App & CORS ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Document Retrieval System API",
    description="Hybrid RAG pipeline for intelligent multi-document Q&A",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://document-retrieval-system-5gqx.onrender.com",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ──────────────────────────────────────────────────────────────
doc_store = EnhancedDocumentStoreHybrid(use_rerank=False)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    filter_type: Optional[str] = None
    auto_route: bool = True
    num_chunks: int = 4
    use_rerank: bool = False


class RerankSettingRequest(BaseModel):
    enabled: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/status")
def get_status():
    """Check whether a document has been loaded and is ready to query."""
    return {"ready": doc_store.is_ready}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and process it through the RAG pipeline.
    Returns extraction statistics on success.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save to a temp file so the existing pipeline can read it from disk
    suffix = ".pdf"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        success, stats = doc_store.process_pdf(
            tmp_path,
            filename=file.filename,
            embed_model=embed_model,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not success:
        raise HTTPException(status_code=500, detail=stats.get("error", "Processing failed."))

    return sanitize({"success": True, "stats": stats})


@app.get("/structure")
def get_structure():
    """Return the document structure (document types, page ranges, chunk counts)."""
    if not doc_store.is_ready:
        raise HTTPException(status_code=400, detail="No document loaded. Upload a PDF first.")
    return sanitize({"structure": doc_store.get_document_structure()})


@app.post("/query")
def query_document(request: QueryRequest):
    """
    Ask a question about the processed document.
    Supports optional filtering by document type, auto-routing, and reranking.
    """
    if not doc_store.is_ready:
        raise HTTPException(status_code=400, detail="No document loaded. Upload a PDF first.")

    filter_type = None if request.filter_type in (None, "All", "") else request.filter_type

    # Apply the rerank setting from the request
    if request.use_rerank != doc_store.use_rerank:
        doc_store.set_rerank(request.use_rerank)

    result = doc_store.query(
        request.question,
        filter_type=filter_type,
        auto_route=request.auto_route and filter_type is None,
        k=request.num_chunks,
        return_details=True,
    )

    return sanitize({
        "answer": result["answer"],
        "confidence": result["confidence"],
        "filter_used": result["filter_used"],
        "sources": result.get("sources", []),
        "retrieval_details": result.get("retrieval_details", {}),
    })


@app.post("/clear")
def clear_store():
    """Clear all stored documents and reset the document store."""
    doc_store.clear()
    return {"success": True, "message": "Document store cleared."}


@app.post("/settings/rerank")
def set_rerank(request: RerankSettingRequest):
    """Enable or disable the BGE Reranker (served via Modal)."""
    doc_store.set_rerank(request.enabled)
    return {"rerank_enabled": request.enabled}


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
