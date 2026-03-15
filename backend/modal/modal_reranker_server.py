"""
modal_reranker_server.py — BGE Reranker-v2-m3 served via Modal

Hosts the BAAI/bge-reranker-v2-m3 CrossEncoder (2GB+) on a Modal T4 GPU,
exposing a FastAPI endpoint POST /rerank that accepts a query + list of texts
and returns float scores.

Deploy:
    modal deploy modal/modal_reranker_server.py

The local retriever.py calls this endpoint when use_rerank=True and
RERANKER_URL is set in the environment.
"""
import modal
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# -----------------------------------------------------------------------
# Image — only needs sentence-transformers + torch
# -----------------------------------------------------------------------
reranker_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers>=2.7.0",
        "torch>=2.2.0",
        "fastapi[standard]",
    )
)

app = modal.App("minilm-reranker")

# Volume to persist model weights
volume = modal.Volume.from_name("minilm-reranker-weights", create_if_missing=True)
MODEL_DIR = "/model_cache"
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# -----------------------------------------------------------------------
# Step 1: Download weights once (CPU, cheap)
# -----------------------------------------------------------------------
@app.function(
    image=reranker_image,
    volumes={MODEL_DIR: volume},
    timeout=1800,               # 30 min — model is ~2GB
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model():
    """Download BGE-reranker-v2-m3 weights to the Modal Volume."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_NAME} to {MODEL_DIR}...")
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=f"{MODEL_DIR}/{MODEL_NAME}",
        ignore_patterns=["*.pt", "*.bin"],   # prefer safetensors
    )
    volume.commit()
    print("✅ Download complete.")


# -----------------------------------------------------------------------
# Step 2: Serve (GPU)
# -----------------------------------------------------------------------
web_app = FastAPI(title="BGE Reranker API")


class RerankRequest(BaseModel):
    query: str
    texts: List[str]


class RerankResponse(BaseModel):
    scores: List[float]
    count: int


@app.function(
    image=reranker_image,
    gpu="T4",
    volumes={MODEL_DIR: volume},
    timeout=120,
)
@modal.asgi_app()
def serve_reranker():
    """FastAPI ASGI server serving the BGE reranker."""
    from sentence_transformers import CrossEncoder

    # Load once at container startup
    import os
    model_path = f"{MODEL_DIR}/{MODEL_NAME}"
    
    if not os.path.isdir(model_path):
        print(f"⚠️ Local model path not found at {model_path}. Falling back to repo name: {MODEL_NAME}")
        model_path = MODEL_NAME

    print(f"🔄 Loading reranker from: {model_path}...")
    model = CrossEncoder(
        model_path,
        max_length=1024,
        device="cuda",
    )
    print("✅ Reranker model loaded.")

    @web_app.post("/rerank", response_model=RerankResponse)
    async def rerank(req: RerankRequest):
        pairs = [[req.query, text] for text in req.texts]
        scores = model.predict(pairs)
        return RerankResponse(scores=scores.tolist(), count=len(scores))

    @web_app.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_NAME}

    return web_app
