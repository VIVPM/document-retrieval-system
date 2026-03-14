"""
gemini_setup.py — LLM and embedding model initialisation.

PRIMARY:  Sarvam-30B via OpenAI-compatible API (https://api.sarvam.ai/v1)
FALLBACK: Gemini 2.5 Flash via native google-genai SDK (commented out below)

To switch back to Gemini:
  1. Comment out the "SARVAM" section
  2. Uncomment the "GEMINI" section
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


# ---------------------------------------------------------------------------
# Shared wrapper — both Sarvam and Gemini expose a `.complete()` method
# returning an object with a `.text` attribute, so all core modules work
# identically regardless of which LLM is active.
# ---------------------------------------------------------------------------
class MockResponse:
    """Minimal response object with a .text attribute (LlamaIndex-compatible)."""
    def __init__(self, text: str):
        self.text = text


# ===========================================================================
# 1-A  PRIMARY: SARVAM-30B  (OpenAI-compatible API)
# ===========================================================================
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if not SARVAM_API_KEY:
    print("⚠️  WARNING: SARVAM_API_KEY not found in .env — LLM will be None.")

try:
    from openai import OpenAI as _OpenAI

    print("🔄 Connecting to Sarvam-30B via sarvam.ai API...")

    _sarvam_client = _OpenAI(
        api_key=SARVAM_API_KEY,
        base_url="https://api.sarvam.ai/v1",
    )

    class SarvamLLM:
        """LlamaIndex-compatible wrapper around Sarvam's OpenAI-compat API."""

        def __init__(self, model: str = "sarvam-105b", temperature: float = 0.3,
                     max_tokens: int = 2048):
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens

        def complete(self, prompt: str, **kwargs) -> MockResponse:
            temp = kwargs.get("temperature", self.temperature)
            max_tok = kwargs.get("max_tokens", self.max_tokens)

            response = _sarvam_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=max_tok,
            )
            return MockResponse((response.choices[0].message.content or "").strip())

    llm = SarvamLLM(model="sarvam-105b")
    print("✅ Connected to Sarvam-105B.")

except ImportError:
    print("⚠️  openai package not installed. Run: pip install openai")
    llm = None
except Exception as e:
    print(f"⚠️  Sarvam connection failed: {e}")
    llm = None


# ===========================================================================
# 1-B  FALLBACK: GEMINI 2.5 FLASH (comment back in if Sarvam is unavailable)
# ===========================================================================
# api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
#
# if not api_key:
#     print("⚠️ WARNING: GEMINI_API_KEY not found in environment or .env file.")
#
# try:
#     from google import genai
#     from google.genai import types
#
#     print("🔄 Connecting to Gemini 2.5 Flash via native google-genai SDK...")
#
#     class LlamaIndexCompatibleGeminiWrapper:
#         def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
#             self.client = genai.Client(api_key=api_key)
#             self.model_name = model_name
#
#         def complete(self, prompt: str, **kwargs):
#             temp = kwargs.get("temperature", 0.3)
#             max_tokens = kwargs.get("max_tokens", 2048)
#             response = self.client.models.generate_content(
#                 model=self.model_name,
#                 contents=prompt,
#                 config=types.GenerateContentConfig(
#                     temperature=temp,
#                     max_output_tokens=max_tokens,
#                 )
#             )
#             return MockResponse(response.text)
#
#     llm = LlamaIndexCompatibleGeminiWrapper(api_key=api_key)
#     print("✅ Connected to Gemini Native SDK.")
# except ImportError:
#     print("⚠️ ERROR: Could not import google-genai.")
#     llm = None


# ===========================================================================
# 2.  EMBEDDING MODEL  (always local — fast and free)
# ===========================================================================
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

print(f"🔄 Loading embedding model: {EMBED_MODEL_NAME}...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print("✅ Embedding model ready.")
