"""
llm_setup.py — LLM and embedding model initialisation.

Loads (once) and exports:
  - gemma_llm       : quantised Gemma-2 9B via HuggingFaceLLM (4-bit NF4)
  - embed_model     : SentenceTransformer BAAI/bge-small-en-v1.5 (for FAISS)
  - llama_embed_model: HuggingFaceEmbedding wrapper (for LlamaIndex chunker)
  - print_gpu_memory(): helper to inspect VRAM usage

All other modules import `gemma_llm` and `embed_model` from here — keeps
model loading centralised and avoids duplicate GPU allocations.
"""

import os
import torch
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Prevent Windows Privilege Errors during Docling model downloads
# Forces Hugging Face to hard-copy files instead of creating symbolic links
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# ---------------------------------------------------------------------------
# Hugging Face authentication
# ---------------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("⚠️ Warning: HF_TOKEN not found in .env file. Model downloads may fail.")

# ---------------------------------------------------------------------------
# GPU memory helper
# ---------------------------------------------------------------------------

def print_gpu_memory() -> None:
    """Print current GPU VRAM allocation and reservation."""
    if not torch.cuda.is_available():
        print("No CUDA device available.")
        return
    print(f"GPU Memory Allocated : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU Memory Reserved  : {torch.cuda.memory_reserved()  / 1024**3:.2f} GB")

# ---------------------------------------------------------------------------
# Gemma-2 9B Hosted on Modal (via OpenAI-compatible vLLM API)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a Q&A assistant. Your goal is to answer questions as accurately "
    "as possible based on the instructions and context provided without "
    "providing any explanations."
)

print("🔄 Connecting to Remote Gemma-2 9B on Modal...")
# URL comes from the modal deploy command
MODAL_API_URL = os.getenv("LLM_URL")
if not MODAL_API_URL:
    raise EnvironmentError(
        "LLM_URL is not set. Add it to your .env file.\n"
        "  LLM_URL=https://<your-deployment>.modal.run/v1"
    )

try:
    gemma_llm = OpenAILike(
        model="google/gemma-2-9b-it",
        api_base=MODAL_API_URL,
        api_key="modal-dummy-key", # Not required by our vLLM instance
        temperature=0.3,
        system_prompt=_SYSTEM_PROMPT,
        max_tokens=2048,
        timeout=300.0, # 5 minutes cold-start allowance for Modal
        max_retries=2,
        additional_kwargs={"stop": ["<end_of_turn>"]} # Gemma 2 specific stop token
    )
    print("✅ Connected to Remote Gemma-2 9B.")
except Exception as e:
    print(f"⚠️ Failed to connect to Modal LLM: {e}")
    gemma_llm = None

# ---------------------------------------------------------------------------
# Embedding models
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

print(f"🔄 Loading embedding model: {EMBED_MODEL_NAME}...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

print("✅ Embedding models ready.")
