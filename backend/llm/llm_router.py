"""
Answer generation and embeddings.

Generation runs on whichever provider LLM_MODEL selects; embeddings are always
Gemini. That asymmetry is not a shortcut: the Pinecone index is built at
EMBED_DIM from gemini-embedding-2, and another provider's vectors would occupy
a different space, so moving embeddings means re-embedding every document
rather than changing a setting.
"""

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from google import genai
from google.genai import types

from logging_setup import get_logger

log = get_logger("drs.llm")

# By explicit path, not cwd: this module raises on an unset LLM_MODEL, and a
# bare load_dotenv() finds nothing when the process starts from the repo
# root, turning "run a script from the wrong directory" into a hard crash.
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

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

# Ceiling on ONE provider call. Without it a hung connection blocks the calling
# thread for ever: an ingest thread stops making progress but keeps its job
# lease, and an answer thread holds a request open with no way for the client to
# learn anything went wrong. Generous, because a long answer with thinking
# legitimately takes tens of seconds -- this bounds the hang, it does not tune
# latency.
LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "120"))

# Whether a failed provider may fall back to the other one. On by default, but
# inert unless the other provider's credentials are also present, so a
# single-provider deployment is unaffected. Set 0 to pin traffic to LLM_MODEL
# even during an outage -- which is what you want if the second provider bills
# a budget you are not willing to spend.
ALLOW_FAILOVER = os.getenv("ALLOW_FAILOVER", "1").strip() != "0"

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


# ── Cost ──────────────────────────────────────────────────────────────────────
# Tokens are not money. Rates differ per model, and THINKING tokens bill at the
# OUTPUT rate -- which for this app is the largest single line on a query's
# bill, because answers run with a 2048-token thinking budget.
#
# Rates are USD per MILLION tokens, checked against the provider pricing pages.
# They will drift; treat a number here as "last verified", not as truth. The
# 4.0x flash/flash-lite ratio these produce matches the repo's own measured
# $/q in the model sweep, which is the sanity check that they are not nonsense.
PRICE_PER_MTOK = {
    "gemini-2.5-flash":       {"in": 0.30, "out": 2.50},
    "gemini-2.5-flash-lite":  {"in": 0.10, "out": 0.40},
    "gemini-3.5-flash-lite":  {"in": 0.30, "out": 2.50},   # "lite" tracks the
                                                           # generation, not the
                                                           # old price point
    "gemini-3.6-flash":       {"in": 1.00, "out": 8.00},
}

# Workers AI bills NEURONS, not tokens, so its cost cannot be derived from the
# token counts. The API returns a neuron count per call; this converts it.
# 10,000 neurons/day are free, and that allowance is per ACCOUNT -- shared with
# anything else on it -- so a cost of 0 here does not mean a call was free, only
# that it came out of an allowance something else may also be spending.
USD_PER_1K_NEURONS = 0.011


def estimate_cost_usd(usage: dict) -> float | None:
    """Cost of one call from its usage dict, or None when it cannot be priced.

    None rather than 0.0 for an unknown model: a zero would quietly under-report
    a real bill and look like a free call, which is worse than an obvious gap.
    """
    if not usage:
        return None

    neurons = usage.get("neurons")
    if neurons:
        return (neurons / 1000.0) * USD_PER_1K_NEURONS

    rates = PRICE_PER_MTOK.get(usage.get("model") or "")
    if rates is None:
        return None

    # Thinking tokens are billed at the OUTPUT rate and are reported SEPARATELY
    # from output tokens, so they have to be added rather than assumed included.
    billed_out = (usage.get("output_tokens") or 0) + (usage.get("thinking_tokens") or 0)
    return ((usage.get("prompt_tokens") or 0) / 1e6 * rates["in"]
            + billed_out / 1e6 * rates["out"])


# ── Resilience ────────────────────────────────────────────────────────────────
# Retry the transport, never the reasoning. A 429 or a 503 is the provider
# saying "not now"; a 400 or a refusal is it saying "not this", and retrying
# that pays twice for the same answer.

RETRY_ATTEMPTS = 3          # total tries, not extra ones
RETRY_BASE_S = 0.5
RETRY_MAX_S = 8.0

# Worth another go: rate limits, overload, gateway and timeout classes.
RETRYABLE_STATUS = frozenset((408, 409, 425, 429, 500, 502, 503, 504))
# Deterministic. The same request will fail the same way, so a retry is pure
# cost -- and on 401/403 it is also a good way to get an account flagged.
TERMINAL_STATUS = frozenset((400, 401, 403, 404, 405, 413, 415, 422))

# Consecutive failures before a provider is considered down, and how long it is
# left alone afterwards. Consecutive rather than a rate, because a rate needs a
# window and a window needs tuning -- five in a row is unambiguous.
BREAKER_THRESHOLD = 5
BREAKER_COOLDOWN_S = 60.0


def _status_of(exc: Exception) -> int | None:
    """HTTP status from either SDK's exception, or None.

    google-genai puts it on .code, the OpenAI client on .status_code. Read by
    attribute rather than by isinstance so this file does not have to import
    both SDKs just to classify an error.
    """
    for attr in ("status_code", "code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _is_retryable(exc: Exception) -> bool:
    """Whether this failure is worth another attempt."""
    status = _status_of(exc)
    if status is not None:
        if status in TERMINAL_STATUS:
            return False
        if status in RETRYABLE_STATUS:
            return True
        return status >= 500          # unknown 5xx: assume transient

    # No status at all means the request never got an answer -- a timeout, a
    # dropped connection, DNS. Those are the transport failures retries exist
    # for. Matched on class NAME so neither SDK has to be imported here.
    name = type(exc).__name__
    return any(k in name for k in ("Timeout", "Connection", "Unavailable", "Socket"))


class _Breaker:
    """One provider's circuit breaker.

    Stops hammering a provider that is down. Without it a queue full of jobs
    burns its whole retry budget against an outage and turns a recoverable
    blip into a pile of permanently failed documents.

    Not thread-safe by lock, deliberately: the worst a race can do is miscount
    a failure by one, and a lock on this path would serialise every LLM call in
    the process for no real benefit.
    """

    def __init__(self, name: str):
        self.name = name
        self.consecutive_failures = 0
        self.opened_at = 0.0

    @property
    def is_open(self) -> bool:
        """True while the provider is being left alone."""
        if self.consecutive_failures < BREAKER_THRESHOLD:
            return False
        if time.monotonic() - self.opened_at >= BREAKER_COOLDOWN_S:
            # Cooldown elapsed: let ONE call through to test the water. It is
            # half-open in effect -- a success resets, a failure re-arms.
            return False
        return True

    def record_success(self) -> None:
        if self.consecutive_failures:
            log.info("provider recovered", extra={"provider": self.name,
                                                  "after_failures": self.consecutive_failures})
        self.consecutive_failures = 0
        self.opened_at = 0.0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures == BREAKER_THRESHOLD:
            self.opened_at = time.monotonic()
            log.error("circuit opened", extra={"provider": self.name,
                                               "failures": self.consecutive_failures,
                                               "cooldown_s": BREAKER_COOLDOWN_S})
        elif self.consecutive_failures > BREAKER_THRESHOLD:
            self.opened_at = time.monotonic()          # re-arm after a failed probe


_breakers: dict[str, _Breaker] = {
    "GEMINI": _Breaker("GEMINI"),
    "CLOUDFLARE": _Breaker("CLOUDFLARE"),
}


def provider_down(provider: str | None = None) -> bool:
    """Is this provider's circuit open? Read by the worker before it claims."""
    return _breakers[(provider or LLM_MODEL)].is_open


def _with_retries(fn, provider: str, what: str):
    """Run fn(), retrying transport failures with jittered backoff.

    Raises the last exception if every attempt fails, so the caller still
    decides what a failure means -- this layer only decides whether to try
    again.
    """
    breaker = _breakers[provider]
    last: Exception | None = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            result = fn()
            breaker.record_success()
            return result
        except Exception as exc:
            last = exc
            breaker.record_failure()
            retryable = _is_retryable(exc)
            log.warning("provider call failed", extra={
                "provider": provider, "op": what, "attempt": attempt,
                "of": RETRY_ATTEMPTS, "status": _status_of(exc),
                "retryable": retryable, "error": f"{type(exc).__name__}: {exc}"[:200]})
            if not retryable or attempt == RETRY_ATTEMPTS:
                raise
            # Exponential with full jitter. Without jitter every in-flight
            # request retries on the same schedule and the second wave lands
            # together, which is what turns a blip into an outage.
            delay = min(RETRY_MAX_S, RETRY_BASE_S * (2 ** (attempt - 1)))
            time.sleep(random.uniform(0, delay))

    raise last                                          # unreachable


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
        # Usage from the most recent stream() call. Streaming yields text,
        # so the counts cannot be returned -- they are left here instead.
        self.last_stream_usage: dict = {}
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
                timeout=LLM_TIMEOUT_S,        # seconds, per the OpenAI client
                max_retries=0,                # retries belong to the queue, not here
            )
            self.label = CLOUDFLARE_MODEL
            print(f"🔄 Cloudflare Workers AI configured ({CLOUDFLARE_MODEL})")
        elif GEMINI_API_KEY:
            # google-genai takes MILLISECONDS here, unlike every other
            # timeout in this file. Passing seconds would set a 120ms
            # deadline and fail every call.
            self._gemini = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options=types.HttpOptions(timeout=LLM_TIMEOUT_S * 1000),
            )
            thinking = ("off" if GEMINI_THINKING_BUDGET == 0
                        else "dynamic" if GEMINI_THINKING_BUDGET < 0
                        else f"{GEMINI_THINKING_BUDGET} tokens")
            print(f"🔄 {GEMINI_CHAT_MODEL} configured (thinking: {thinking})")
            self.label = GEMINI_CHAT_MODEL
        else:
            print("⚠️  GEMINI_API_KEY not set — answer generation unavailable")
            self.label = "No LLM configured"

    def _other_provider(self) -> str | None:
        """The provider to fail over to, or None if it is not usable.

        None is the common case and the safe one: failover needs the OTHER
        provider's credentials present. Building it lazily means a deployment
        configured for one provider pays nothing for this and behaves exactly
        as it did before.
        """
        other = "CLOUDFLARE" if LLM_MODEL == "GEMINI" else "GEMINI"
        if other == "GEMINI":
            return "GEMINI" if GEMINI_API_KEY else None
        return "CLOUDFLARE" if (CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN) else None

    def _client_for(self, provider: str):
        """Return (and build on first use) the client for one provider."""
        if provider == "GEMINI":
            if self._gemini is None:
                self._gemini = genai.Client(
                    api_key=GEMINI_API_KEY,
                    http_options=types.HttpOptions(timeout=LLM_TIMEOUT_S * 1000),
                )
                log.info("built failover client", extra={"provider": "GEMINI"})
            return self._gemini

        if self._cf is None:
            from openai import OpenAI
            self._cf = OpenAI(
                api_key=CLOUDFLARE_API_TOKEN,
                base_url=f"https://api.cloudflare.com/client/v4/accounts/"
                         f"{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
                timeout=LLM_TIMEOUT_S,
                max_retries=0,        # retries are _with_retries' job, not a nested client's
            )
            log.info("built failover client", extra={"provider": "CLOUDFLARE"})
        return self._cf

    def _complete_on(self, provider: str, prompt: str, temp: float, max_tok: int,
                     thinking: int, model: str) -> tuple[str, dict]:
        """One completion on one named provider. Raises; retries live above."""
        self._client_for(provider)
        if provider == "CLOUDFLARE":
            return self._cloudflare_complete(prompt, temp, max_tok)
        return self._gemini_complete(prompt, temp, max_tok, thinking, model)

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
            # Workers AI bills neurons, not tokens; without this the call
            # cannot be priced at all.
            "neurons": getattr(u, "neurons", None) or 0,
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

        primary = LLM_MODEL
        order = [primary]
        if ALLOW_FAILOVER and self._other_provider() is not None:
            order.append(self._other_provider())

        last_error: Exception | None = None
        for provider in order:
            if provider != primary:
                log.warning("failing over", extra={"from": primary, "to": provider})
            if _breakers[provider].is_open:
                log.warning("skipping provider, circuit open", extra={"provider": provider})
                continue
            try:
                text, usage = _with_retries(
                    lambda p=provider: self._complete_on(p, prompt, temp, max_tok,
                                                         thinking, model),
                    provider, "complete")
                if text:
                    # Cost per call, tagged with the correlation id. Summing by
                    # request_id then answers "what did this cost" for a message
                    # AND for an ingest -- which a span cannot, because ingest
                    # runs its LLM calls in raw thread pools that do not carry
                    # the tracing context.
                    cost = estimate_cost_usd(usage)
                    log.info("llm call", extra={
                        "provider": provider, "model": usage.get("model"),
                        "in": usage.get("prompt_tokens"), "out": usage.get("output_tokens"),
                        "thinking": usage.get("thinking_tokens"),
                        "cost_usd": None if cost is None else round(cost, 6)})
                    return MockResponse(text, usage)
                # An empty completion is not a transport failure, so it does not
                # trip the breaker and must not fail over -- both providers
                # would return the same nothing for a prompt the model will not
                # answer. answer_generator turns this into an explicit error.
                return MockResponse("", usage)
            except Exception as e:
                last_error = e

        if last_error is not None:
            log.error("all providers failed", extra={
                "tried": order, "error": f"{type(last_error).__name__}: {last_error}"[:200]})
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

        # Cleared per stream: reading a previous answer's usage would misreport
        # cost silently, which is worse than reporting none.
        self.last_stream_usage = {}

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
                    u = getattr(chunk, "usage", None)
                    if u is not None:
                        self.last_stream_usage = {
                            "model": CLOUDFLARE_MODEL,
                            "prompt_tokens": getattr(u, "prompt_tokens", None) or 0,
                            "output_tokens": getattr(u, "completion_tokens", None) or 0,
                            "thinking_tokens": 0,
                            "cached_tokens": 0,
                            "neurons": getattr(u, "neurons", None) or 0,
                        }
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
                # The final chunk carries usage_metadata, so a streamed answer
                # CAN be priced -- it just has to be picked up on the way past.
                # last_stream_usage is how the caller reads it afterwards.
                m = getattr(chunk, "usage_metadata", None)
                if m is not None:
                    self.last_stream_usage = {
                        "model": model,
                        "prompt_tokens": getattr(m, "prompt_token_count", None) or 0,
                        "output_tokens": getattr(m, "candidates_token_count", None) or 0,
                        "thinking_tokens": getattr(m, "thoughts_token_count", None) or 0,
                        "cached_tokens": getattr(m, "cached_content_token_count", None) or 0,
                    }
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"⚠️  {model} stream failed ({type(e).__name__}: {e})")


class GeminiEmbeddingModel:
    """Wrapper to make Gemini embeddings drop-in compatible with SentenceTransformers."""
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in .env for embeddings.")
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=LLM_TIMEOUT_S * 1000),
        )

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
