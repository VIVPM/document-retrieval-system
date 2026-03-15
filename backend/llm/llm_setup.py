"""
llm_setup.py — LLM initialisation (Legacy).

Loads (once) and exports:
  - gemma_llm       : quantised Gemma-2 9B (now usually handled by llm_router.py)
  - print_gpu_memory(): helper to inspect VRAM usage

Note: This file is largely superseded by llm_router.py.
"""

import os
import torch
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Prevent Windows Privilege Errors during Docling model downloads
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
# Gemma-2 9B Hosted on Modal (Legacy reference)
# ---------------------------------------------------------------------------

print("🔄 Note: llm_setup.py is legacy. Use llm_router.py for primary access.")

gemma_llm = None # Placeholder to avoid import errors in legacy code
embed_model = None
llama_embed_model = None
