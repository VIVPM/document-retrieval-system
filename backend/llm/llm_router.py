"""
Answer generation and embeddings, both via Gemini.
"""

import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_CHAT_MODEL  = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

GEMINI_FAST_MODEL  = os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash-lite")

GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "2048"))
GEMINI_MAX_OUTPUT  = int(os.getenv("GEMINI_MAX_OUTPUT", "8192"))

GEMINI_EMBED_MODEL = "models/gemini-embedding-2"
EMBED_DIM          = 768
EMBED_CONCURRENCY  = int(os.getenv("EMBED_CONCURRENCY", "8"))


class MockResponse:
    """A completion and its prompt/output/thinking token counts."""

    def __init__(self, text: str, usage: dict | None = None):
        self.text = text
        self.usage = usage or {}


class LLMRouter:
    """Generates completions via Gemini."""

    def __init__(self):
        self._gemini = None
        if GEMINI_API_KEY:
            self._gemini = genai.Client(api_key=GEMINI_API_KEY)
            thinking = ("off" if GEMINI_THINKING_BUDGET == 0
                        else "dynamic" if GEMINI_THINKING_BUDGET < 0
                        else f"{GEMINI_THINKING_BUDGET} tokens")
            print(f"🔄 {GEMINI_CHAT_MODEL} configured (thinking: {thinking})")
        else:
            print("⚠️  GEMINI_API_KEY not set — answer generation unavailable")
        self.label = GEMINI_CHAT_MODEL if self._gemini else "No LLM configured"

    def _gemini_complete(self, prompt: str, temperature: float, max_tokens: int,
                         thinking_budget: int, model: str) -> tuple[str, dict]:
        """Single Gemini call; returns (text, usage-metadata dict)."""
        response = self._gemini.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=thinking_budget),
            ),
        )
        text = (response.text or "").strip()

        if not text:
            cand = (response.candidates or [None])[0]
            finish = getattr(cand, "finish_reason", None) if cand else None
            print(f"   ↳ {model} returned no text "
                  f"(finish={finish!r}, "
                  f"feedback={getattr(response, 'prompt_feedback', None)!r})")

        m = getattr(response, "usage_metadata", None)
        usage = {
            "model": model,
            "prompt_tokens": getattr(m, "prompt_token_count", None) or 0,
            "output_tokens": getattr(m, "candidates_token_count", None) or 0,
            "thinking_tokens": getattr(m, "thoughts_token_count", None) or 0,
            "cached_tokens": getattr(m, "cached_content_token_count", None) or 0,
        }
        return text, usage

    def complete(self, prompt: str, **kwargs) -> MockResponse:
        """
        Generate a completion.

        `fast=True` routes to GEMINI_FAST_MODEL; `model` overrides both.
        `thinking_budget` and `temperature` override the module defaults.
        """
        temp     = kwargs.get("temperature", 0.3)
        max_tok  = kwargs.get("max_tokens", GEMINI_MAX_OUTPUT)
        thinking = kwargs.get("thinking_budget", GEMINI_THINKING_BUDGET)
        model    = kwargs.get("model") or (GEMINI_FAST_MODEL if kwargs.get("fast")
                                           else GEMINI_CHAT_MODEL)

        if self._gemini:
            try:
                text, usage = self._gemini_complete(
                    prompt, temp, max_tok, thinking, model)
                if text:
                    return MockResponse(text, usage)
            except Exception as e:
                print(f"⚠️  {model} failed ({type(e).__name__}: {e})")

        return MockResponse("")

    def stream(self, prompt: str, **kwargs):
        """
        Yield answer text chunks as Gemini produces them.

        Same config as complete() — capped thinking, generous max_output — so a
        streamed answer is identical to the buffered one, just delivered token by
        token. Gemini's stream is a blocking sync generator; the SSE endpoint
        pumps it through asyncio.to_thread so it never blocks the event loop.
        """
        if not self._gemini:
            return
        temp     = kwargs.get("temperature", 0.3)
        max_tok  = kwargs.get("max_tokens", GEMINI_MAX_OUTPUT)
        thinking = kwargs.get("thinking_budget", GEMINI_THINKING_BUDGET)
        model    = kwargs.get("model") or GEMINI_CHAT_MODEL
        try:
            for chunk in self._gemini.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temp,
                    max_output_tokens=max_tok,
                    thinking_config=types.ThinkingConfig(thinking_budget=thinking),
                ),
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"⚠️  {model} stream failed ({type(e).__name__}: {e})")


class GeminiEmbeddingModel:
    """Wrapper to make Gemini embeddings drop-in compatible with SentenceTransformers."""
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in .env for embeddings.")
        self.client = genai.Client(api_key=api_key)

    def encode(self, texts: list[str], show_progress_bar: bool = False,
               task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """
        Embed texts with Gemini.

        task_type must be RETRIEVAL_DOCUMENT when embedding chunks for the
        index and RETRIEVAL_QUERY when embedding a search query — Gemini
        produces asymmetric embeddings and using the document type for
        queries measurably degrades similarity.

        EMBED_DIM is a Matryoshka truncation of the model's native 3072,
        which the model is trained to support.
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []

        if len(texts) == 1:
            return [self._embed_one(texts[0], task_type)]

        with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as pool:
            return list(pool.map(lambda t: self._embed_one(t, task_type), texts))

    def _embed_one(self, text: str, task_type: str) -> list[float]:
        """Embed one string via Gemini and validate the returned dimension."""
        result = self.client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBED_DIM,
            )
        )
        if not result.embeddings or not result.embeddings[0].values:
            raise RuntimeError(
                f"Gemini returned no embedding for a {len(text)}-char text "
                f"(task_type={task_type})."
            )
        values = list(result.embeddings[0].values)
        if len(values) != EMBED_DIM:
            raise RuntimeError(
                f"Gemini returned {len(values)} dimensions, expected "
                f"{EMBED_DIM}."
            )
        return values


llm = LLMRouter()
print(f"🟢 LLM Router ready: {llm.label}")

print(f"🔄 Loading cloud embedding model: {GEMINI_EMBED_MODEL}...")
try:
    embed_model = GeminiEmbeddingModel(api_key=GEMINI_API_KEY)
    print("✅ Embedding model ready.")
except ValueError as e:
    print(f"⚠️ {e}")
    embed_model = None
