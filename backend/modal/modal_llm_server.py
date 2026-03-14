"""
modal_llm_server.py — Gemma-2 9B (FP8 quantized) via vLLM on Modal

Follows the official Modal vLLM deployment pattern exactly:
  https://modal.com/blog/how-to-deploy-vllm

Deploy:
    # 1. Download weights once (CPU only, cheap)
    modal run modal/modal_llm_server.py::download_model

    # 2. Deploy the live server
    modal deploy modal/modal_llm_server.py

    # 3. Copy the printed URL into .env as LLM_URL
"""
import subprocess
import modal

# ── Model config ──────────────────────────────────────────────────────────────
# FP8-quantized Gemma-2 9B by Neural Magic — ~9GB vs ~18GB for full BF16 model
MODEL_NAME = "neuralmagic/gemma-2-9b-it-FP8"
MODEL_REVISION = None          # pin to a specific commit hash for reproducibility
                               # e.g. "abc123..." — leave None to use latest
SERVED_MODEL_NAME = "google/gemma-2-9b-it"   # name the OpenAI client sends
MODEL_DIR = "/model_cache"
N_GPU = 1

MINUTES = 60

# ── FAST_BOOT ─────────────────────────────────────────────────────────────────
# True  → --enforce-eager: skip CUDA graph compilation → faster cold starts
# False → --no-enforce-eager: full CUDA graphs → better sustained throughput
FAST_BOOT = True

# ── Image (exact versions from the official Modal blog) ───────────────────────
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])                          # clear default CUDA entrypoint
    .uv_pip_install(                         # uv is faster than pip
        "vllm==0.10.2",
        "torch==2.8.0",
        "flashinfer-python==0.3.1",
        "huggingface_hub[hf_transfer]==0.35.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("gemma-2-9b-vllm-server")

# Volume caches the ~9GB weights so cold starts skip re-downloading
volume = modal.Volume.from_name("gemma-9b-weights", create_if_missing=True)


# ── Step 1: Download weights (CPU only — run once) ────────────────────────────
@app.function(
    image=vllm_image,
    volumes={MODEL_DIR: volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model():
    """Download FP8-quantized Gemma-2 9B weights to the Modal Volume."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_NAME} to {MODEL_DIR}...")
    kwargs = dict(
        repo_id=MODEL_NAME,
        local_dir=f"{MODEL_DIR}/{MODEL_NAME}",
        ignore_patterns=["*.pt", "*.bin"],   # prefer safetensors
    )
    if MODEL_REVISION:
        kwargs["revision"] = MODEL_REVISION
    snapshot_download(**kwargs)
    volume.commit()
    print("✅ Download complete.")


# ── Step 2: Serve (GPU) ───────────────────────────────────────────────────────
@app.function(
    image=vllm_image,
    gpu=f"L4:{N_GPU}",
    volumes={MODEL_DIR: volume},
    timeout=10 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=32) # queue up to 32 requests per replica
@modal.web_server(port=8000, startup_timeout=10 * MINUTES)
def serve():
    """
    Launch vLLM as an OpenAI-compatible background server.
    llm_setup.py uses OpenAILike which works with this endpoint as-is.
    """
    cmd = [
        "vllm", "serve",
        "--uvicorn-log-level=info",
        f"{MODEL_DIR}/{MODEL_NAME}",
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "8192",
        "--tensor-parallel-size", str(N_GPU),
        "--dtype", "bfloat16",                      # Gemma-2 requires bfloat16 (float16 unsupported)
    ]

    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]

    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    print(f"🚀 Starting vLLM: {' '.join(cmd)}")
    subprocess.Popen(cmd)
