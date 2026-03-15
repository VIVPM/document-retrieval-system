"""
llm_router.py — Smart LLM loader with automatic per-request fallback.

Strategy:
  1. PRIMARY  : Modal vLLM server (Gemma-2 9B FP8) — if LLM_URL is set
  2. FALLBACK : Sarvam-105B via sarvam.ai API — used per-request if Modal fails

No upfront health check — Modal cold starts take time but do connect.
Fallback happens silently if a specific .complete() call fails.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

LLM_URL            = os.getenv("LLM_URL", "").strip()
SARVAM_KEY         = os.getenv("SARVAM_API_KEY", "").strip()
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "").strip()
MODAL_KEY          = "modal-dummy-key"
MODAL_MODEL        = "google/gemma-2-9b-it"
SARVAM_MODEL       = "sarvam-105b"
SARVAM_URL         = "https://api.sarvam.ai/v1"
GEMINI_EMBED_MODEL = "models/gemini-embedding-001"


class MockResponse:
    def __init__(self, text: str):
        self.text = text


class LLMRouter:
    """
    Tries Modal LLM first on every .complete() call.
    If Modal fails (cold start timeout, error, etc.), silently falls back
    to Sarvam-105B for that request only.
    """

    def __init__(self):
        from openai import OpenAI

        self._modal_client = None
        self._sarvam_client = None

        if LLM_URL:
            self._modal_client = OpenAI(api_key=MODAL_KEY, base_url=LLM_URL)
            print(f"🔄 Modal LLM configured: {LLM_URL}")
        else:
            print("ℹ️  LLM_URL not set — Modal LLM disabled")

        if SARVAM_KEY:
            self._sarvam_client = OpenAI(api_key=SARVAM_KEY, base_url=SARVAM_URL)
            print(f"🔄 Sarvam-105B configured as fallback")
        else:
            print("⚠️  SARVAM_API_KEY not set — no fallback available")

        self.label = self._describe()

    def _describe(self) -> str:
        if self._modal_client and self._sarvam_client:
            return "Modal/Gemma-2-9B → Sarvam-105B fallback"
        elif self._modal_client:
            return "Modal/Gemma-2-9B (no fallback)"
        elif self._sarvam_client:
            return "Sarvam-105B (no Modal URL set)"
        return "No LLM configured"

    def _chat(self, client, model: str, prompt: str,
              temperature: float, max_tokens: int) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    def complete(self, prompt: str, **kwargs) -> MockResponse:
        temp     = kwargs.get("temperature", 0.3)
        max_tok  = kwargs.get("max_tokens", 2048)

        # Try Modal first
        if self._modal_client:
            try:
                text = self._chat(self._modal_client, MODAL_MODEL,
                                  prompt, temp, max_tok)
                return MockResponse(text)
            except Exception as e:
                print(f"⚠️  Modal LLM failed ({type(e).__name__}: {e}) — falling back to Sarvam")

        # Fallback to Sarvam
        if self._sarvam_client:
            try:
                text = self._chat(self._sarvam_client, SARVAM_MODEL,
                                  prompt, temp, max_tok)
                return MockResponse(text)
            except Exception as e:
                print(f"❌ Sarvam fallback also failed: {e}")

        return MockResponse("")


class GeminiEmbeddingModel:
    """Wrapper to make Gemini embeddings drop-in compatible with SentenceTransformers."""
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in .env for embeddings.")
        self.client = genai.Client(api_key=api_key)

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> list[list[float]]:
        # Handle single string
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        # Batching might be needed for very large lists, but Gemini handles 
        # reasonable array inputs natively. For safety with large docs, we loop.
        for text in texts:
            try:
                result = self.client.models.embed_content(
                    model=GEMINI_EMBED_MODEL,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=1024
                    )
                )
                embeddings.append(list(result.embeddings[0].values))
            except Exception as e:
                print(f"❌ Gemini embedding error: {e}")
                # Fallback zero vector on failure to not break pipeline
                embeddings.append([0.0] * 1024)
        return embeddings


# ── Module-level singletons (imported by all core modules) ──────────────────
llm = LLMRouter()
print(f"🟢 LLM Router ready: {llm.label}")

# ── Embedding model (Cloud) ─────────────────────────────────────────────────
print(f"🔄 Loading cloud embedding model: {GEMINI_EMBED_MODEL}...")
try:
    embed_model = GeminiEmbeddingModel(api_key=GEMINI_API_KEY)
    print("✅ Embedding model ready.")
except ValueError as e:
    print(f"⚠️ {e}")
    embed_model = None

