"""
Answer generation and embeddings.

Generation runs on whichever provider LLM_MODEL selects; embeddings are always
Gemini. That asymmetry is not a shortcut: the Pinecone index is built at
EMBED_DIM from gemini-embedding-2, and another provider's vectors would occupy
a different space, so moving embeddings means re-embedding every document
rather than changing a setting.
"""

import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Which provider generates text: "GEMINI" or "CLOUDFLARE" (Workers AI, through
# its OpenAI-compatible endpoint). Required with no default — this picks which
# account gets billed, and a fallback would quietly send real traffic to a
# provider nobody chose.
LLM_MODEL = (os.getenv("LLM_MODEL") or "").strip().upper()
_PROVIDERS = ("GEMINI", "CLOUDFLARE")
if LLM_MODEL not in _PROVIDERS:
    raise RuntimeError(
        f"LLM_MODEL must be one of {_PROVIDERS}, got {LLM_MODEL or '<unset>'!r}. "
        "Set it in backend/.env and in the environment of whatever hosts this."
    )

GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_CHAT_MODEL  = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

GEMINI_FAST_MODEL  = os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash-lite")

GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "2048"))
GEMINI_MAX_OUTPUT  = int(os.getenv("GEMINI_MAX_OUTPUT", "8192"))

# One model serves both tiers. Chosen by measurement, not price: it is the only
# candidate that is NOT a reasoning model, and reasoning models return
# content=None at this repo's tight budgets (boundary detection runs at
# max_tokens=8). Measured on the real prompts — answers 10/10, the rewriter's
# meta-question check 8/8, classification and boundary both clean; gpt-oss-20b
# and qwen3-30b each returned null on boundary detection. See the roadmap
# before swapping it.
CLOUDFLARE_MODEL      = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
CLOUDFLARE_API_TOKEN  = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()

GEMINI_EMBED_MODEL = "models/gemini-embedding-2"
EMBED_DIM          = 768
EMBED_CONCURRENCY  = int(os.getenv("EMBED_CONCURRENCY", "8"))


class MockResponse:
    """A completion and its prompt/output/thinking token counts."""

    def __init__(self, text: str, usage: dict | None = None):
        self.text = text
        self.usage = usage or {}


class LLMRouter:
    """Generates completions via whichever provider LLM_MODEL selects."""

    def __init__(self):
        self._gemini = None
        self._cf = None
        if LLM_MODEL == "CLOUDFLARE":
            missing = [n for n, v in (("CLOUDFLARE_ACCOUNT_ID", CLOUDFLARE_ACCOUNT_ID),
                                      ("CLOUDFLARE_API_TOKEN", CLOUDFLARE_API_TOKEN)) if not v]
            if missing:
                raise RuntimeError(
                    f"LLM_MODEL=CLOUDFLARE needs {' and '.join(missing)}. "
                    "Set them in backend/.env."
                )
            from openai import OpenAI
            self._cf = OpenAI(
                api_key=CLOUDFLARE_API_TOKEN,
                base_url=f"https://api.cloudflare.com/client/v4/accounts/"
                         f"{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
            )
            self.label = CLOUDFLARE_MODEL
            print(f"🔄 Cloudflare Workers AI configured ({CLOUDFLARE_MODEL})")
        elif GEMINI_API_KEY:
            self._gemini = genai.Client(api_key=GEMINI_API_KEY)
            thinking = ("off" if GEMINI_THINKING_BUDGET == 0
                        else "dynamic" if GEMINI_THINKING_BUDGET < 0
                        else f"{GEMINI_THINKING_BUDGET} tokens")
            print(f"🔄 {GEMINI_CHAT_MODEL} configured (thinking: {thinking})")
            self.label = GEMINI_CHAT_MODEL
        else:
            print("⚠️  GEMINI_API_KEY not set — answer generation unavailable")
            self.label = "No LLM configured"

    def _cloudflare_complete(self, prompt: str, temperature: float,
                             max_tokens: int) -> tuple[str, dict]:
        """Single Workers AI call through the OpenAI-compatible endpoint.

        No thinking_budget equivalent: the chosen model does not reason, which is
        why the caller's max_tokens can stay as tight as it is for Gemini.
        """
        r = self._cf.chat.completions.create(
            model=CLOUDFLARE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = r.choices[0].message.content
        text = ("" if content is None else str(content)).strip()
        if not text:
            print(f"   ↳ {CLOUDFLARE_MODEL} returned no text "
                  f"(finish={r.choices[0].finish_reason!r})")
        u = getattr(r, "usage", None)
        usage = {
            "model": CLOUDFLARE_MODEL,
            "prompt_tokens": getattr(u, "prompt_tokens", None) or 0,
            "output_tokens": getattr(u, "completion_tokens", None) or 0,
            "thinking_tokens": 0,
            "cached_tokens": 0,
        }
        return text, usage

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

        Both are Gemini-only knobs. Workers AI publishes one model for this job,
        so both tiers point at it there and `fast`/`thinking_budget` are ignored
        — the two-tier split is a cost optimisation, not something callers depend
        on for correctness.
        """
        temp     = kwargs.get("temperature", 0.3)
        max_tok  = kwargs.get("max_tokens", GEMINI_MAX_OUTPUT)
        thinking = kwargs.get("thinking_budget", GEMINI_THINKING_BUDGET)
        model    = kwargs.get("model") or (GEMINI_FAST_MODEL if kwargs.get("fast")
                                           else GEMINI_CHAT_MODEL)

        if self._cf:
            try:
                text, usage = self._cloudflare_complete(prompt, temp, max_tok)
                if text:
                    return MockResponse(text, usage)
            except Exception as e:
                print(f"⚠️  {CLOUDFLARE_MODEL} failed ({type(e).__name__}: {e})")
            return MockResponse("")

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
        Yield answer text chunks as the active provider produces them.

        Same config as complete(), so a streamed answer is identical to the
        buffered one, just delivered token by token. Both providers' streams are
        blocking sync generators; the SSE endpoint pumps them through
        asyncio.to_thread so neither blocks the event loop.
        """
        temp     = kwargs.get("temperature", 0.3)
        max_tok  = kwargs.get("max_tokens", GEMINI_MAX_OUTPUT)
        thinking = kwargs.get("thinking_budget", GEMINI_THINKING_BUDGET)
        model    = kwargs.get("model") or GEMINI_CHAT_MODEL

        if self._cf:
            try:
                for chunk in self._cf.chat.completions.create(
                    model=CLOUDFLARE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=max_tok,
                    stream=True,
                ):
                    if not chunk.choices:
                        continue
                    # Cloudflare's shim sends a chunk that is only a number as a
                    # JSON number, so content arrives as int. The SSE endpoint
                    # joins these into one string and would raise on it.
                    delta = chunk.choices[0].delta.content
                    if delta is not None and delta != "":
                        yield delta if isinstance(delta, str) else str(delta)
            except Exception as e:
                print(f"⚠️  {CLOUDFLARE_MODEL} stream failed ({type(e).__name__}: {e})")
            return

        if not self._gemini:
            return
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
