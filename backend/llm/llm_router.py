"""
Answer generation and embeddings, both via Gemini.

Modal/Gemma-2-9B still deploys and can be re-enabled as a fallback with
USE_MODAL_LLM=1. It is off by default; upgrade_roadmap.txt item 21 covers why.
"""

import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "").strip()

# ── Answer generation ────────────────────────────────────────────────────────
GEMINI_CHAT_MODEL  = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

# Classification and per-page boundary detection run here; answers and query
# rewriting stay on GEMINI_CHAT_MODEL. See CLAUDE.md for why the rewriter is
# not on this one.
GEMINI_FAST_MODEL  = os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash-lite")

# -1 dynamic, 0 off, >0 fixed ceiling. Capped rather than dynamic because
# thinking bills at the output rate and comes out of GEMINI_MAX_OUTPUT.
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "2048"))
GEMINI_MAX_OUTPUT  = int(os.getenv("GEMINI_MAX_OUTPUT", "8192"))

# ── Dormant: Modal / Gemma-2-9B ──────────────────────────────────────────────
# Off by default; the deploy script is still maintained.
USE_MODAL_LLM      = os.getenv("USE_MODAL_LLM", "0") == "1"
LLM_URL            = os.getenv("LLM_URL", "").strip()
MODAL_KEY          = "modal-dummy-key"
MODAL_MODEL        = "google/gemma-2-9b-it"
MODAL_TIMEOUT      = float(os.getenv("MODAL_LLM_TIMEOUT", "45"))
# Must match --max-model-len in modal/modal_llm_server.py; prompt + completion
# has to fit inside it or vLLM rejects the request outright.
MODAL_MAX_CTX      = int(os.getenv("MODAL_MAX_CTX", "4096"))
RESERVE_TOKENS     = 96      # chat template + tokeniser drift safety margin
MIN_ANSWER_TOKENS  = 256     # below this an answer is not worth attempting

GEMINI_EMBED_MODEL = "models/gemini-embedding-2"
# Must equal the Pinecone index dimension, which is immutable. Checked at
# startup by HybridRetriever._assert_index_compatible.
EMBED_DIM          = 768
# Concurrent embed_content calls, NOT a batch size. gemini-embedding-2 cannot
# batch on the Developer API: models.embed_content special-cases this model and
# runs t_contents() over a list, which collapses N strings into ONE multi-part
# content and returns a single vector. Concurrency is the available win.
EMBED_CONCURRENCY  = int(os.getenv("EMBED_CONCURRENCY", "8"))


class MockResponse:
    """A completion and its prompt/output/thinking token counts."""

    def __init__(self, text: str, usage: dict | None = None):
        self.text = text
        self.usage = usage or {}


class LLMRouter:
    """Generates completions, with Modal/Gemma-2-9B as an opt-in fallback."""

    def __init__(self):
        self._gemini = None
        self._modal_client = None

        if GEMINI_API_KEY:
            self._gemini = genai.Client(api_key=GEMINI_API_KEY)
            thinking = ("off" if GEMINI_THINKING_BUDGET == 0
                        else "dynamic" if GEMINI_THINKING_BUDGET < 0
                        else f"{GEMINI_THINKING_BUDGET} tokens")
            print(f"🔄 {GEMINI_CHAT_MODEL} configured (thinking: {thinking})")
        else:
            print("⚠️  GEMINI_API_KEY not set — answer generation unavailable")

        if USE_MODAL_LLM and LLM_URL:
            from openai import OpenAI
            # max_retries=0: retries would multiply the cold-start wait.
            self._modal_client = OpenAI(
                api_key=MODAL_KEY, base_url=LLM_URL,
                timeout=MODAL_TIMEOUT, max_retries=0,
            )
            print(f"🔄 Modal fallback enabled: {LLM_URL} (timeout {MODAL_TIMEOUT}s)")

        self.label = self._describe()

    def _describe(self) -> str:
        if self._gemini and self._modal_client:
            return f"{GEMINI_CHAT_MODEL} → Modal/Gemma-2-9B fallback"
        if self._gemini:
            return GEMINI_CHAT_MODEL
        if self._modal_client:
            return "Modal/Gemma-2-9B (no Gemini key)"
        return "No LLM configured"

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Approximate token count, for sizing a safety margin."""
        try:
            import tiktoken
            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:
            return len(text) // 3        # deliberately pessimistic fallback

    def _budget(self, prompt: str, requested: int, max_ctx: int) -> int:
        """Clamp max_tokens so prompt + completion fits the model's window."""
        if not max_ctx:
            return requested
        room = max_ctx - self._count_tokens(prompt) - RESERVE_TOKENS
        return min(requested, room)

    def _chat(self, client, model: str, prompt: str,
              temperature: float, max_tokens: int) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        text = (choice.message.content or "").strip()

        if not text:
            # Separates "ran out of budget mid-reasoning" from "said nothing".
            reasoning = (getattr(choice.message, "model_extra", None) or {}).get(
                "reasoning_content") or ""
            used = getattr(response.usage, "completion_tokens", "?")
            print(f"   ↳ {model} returned empty content "
                  f"(finish={choice.finish_reason!r}, completion_tokens={used}, "
                  f"reasoning={len(reasoning)} chars, max_tokens={max_tokens})")

        return text

    def _gemini_complete(self, prompt: str, temperature: float, max_tokens: int,
                         thinking_budget: int, model: str) -> tuple[str, dict]:
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
            # Never let an empty completion pass as an answer — say why instead.
            cand = (response.candidates or [None])[0]
            finish = getattr(cand, "finish_reason", None) if cand else None
            print(f"   ↳ {model} returned no text "
                  f"(finish={finish!r}, "
                  f"feedback={getattr(response, 'prompt_feedback', None)!r})")

        m = getattr(response, "usage_metadata", None)
        usage = {
            "model": model,
            "prompt_tokens": getattr(m, "prompt_token_count", None) or 0,
            # Excludes thinking, which Gemini reports separately.
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

        # Opt-in fallback, off unless USE_MODAL_LLM=1.
        if self._modal_client:
            budget = self._budget(prompt, max_tok, MODAL_MAX_CTX)
            if budget < MIN_ANSWER_TOKENS:
                print(f"ℹ️  Prompt leaves only {budget} tokens in Modal's "
                      f"{MODAL_MAX_CTX}-token window — skipping it.")
            else:
                try:
                    text = self._chat(self._modal_client, MODAL_MODEL,
                                      prompt, temp, budget)
                    if text:
                        return MockResponse(text)
                except Exception as e:
                    print(f"⚠️  Modal fallback failed ({type(e).__name__}: {e})")

        return MockResponse("")

    def stream(self, prompt: str, **kwargs):
        """
        Yield answer text chunks as Gemini produces them.

        Same config as complete() — capped thinking, generous max_output — so a
        streamed answer is identical to the buffered one, just delivered token by
        token. Gemini's stream is a blocking sync generator; the SSE endpoint
        pumps it through asyncio.to_thread so it never blocks the event loop.
        The Modal fallback has no streaming path and is skipped here.
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
        result = self.client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                # EMBED_DIM, never a literal — a second number here drifts
                # from the startup guard and fails at upsert instead.
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

