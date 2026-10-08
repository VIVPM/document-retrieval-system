"""OpenTelemetry LLM tracing to Langfuse."""
import base64
import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

_llm_provider = None
_llm_tracer = None


def _have_langfuse() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _resource():
    from opentelemetry.sdk.resources import Resource

    return Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME", "document-retrieval-system"),
        "service.namespace": "document-retrieval-system",
        "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development"),
    })


def init_observability():
    """Instrument provider calls and export LLM spans to Langfuse."""
    global _llm_provider, _llm_tracer
    if not _have_langfuse():
        logger.info("LLM tracing disabled (no Langfuse env).")
        return
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider = TracerProvider(resource=_resource())
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
        creds = f'{os.environ["LANGFUSE_PUBLIC_KEY"]}:{os.environ["LANGFUSE_SECRET_KEY"]}'
        auth = base64.b64encode(creds.encode()).decode()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
            endpoint=f"{host}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}"},
        )))

        from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
        GoogleGenAIInstrumentor().instrument(tracer_provider=provider)
        if os.getenv("LLM_MODEL", "").strip().upper() == "CLOUDFLARE":
            from openinference.instrumentation.openai import OpenAIInstrumentor
            OpenAIInstrumentor().instrument(tracer_provider=provider)
        _llm_provider = provider
        _llm_tracer = provider.get_tracer("chat")
        logger.info("LLM tracing enabled via OTLP: Langfuse (%s)", host)
    except Exception:
        logger.exception("LLM tracing init failed — continuing without it.")


@contextmanager
def trace_message(question: str, user_id, session_id):
    """Wrap one message so its LLM calls share a trace. Yields the span, or None if off."""
    if _llm_tracer is None:
        yield None
        return
    try:
        with _llm_tracer.start_as_current_span("chat-message") as span:
            span.set_attribute("langfuse.trace.name", "chat-message")
            span.set_attribute("langfuse.user.id", str(user_id))
            span.set_attribute("langfuse.session.id", str(session_id))
            span.set_attribute("input.value", question)
            yield span
    except Exception as e:
        logger.warning("trace_message failed — continuing untraced: %s", e)
        yield None


def set_output(span, text: str):
    """Attach the final answer to the message span."""
    if span is None:
        return
    try:
        span.set_attribute("output.value", text)
    except Exception as e:
        logger.debug("set_output failed: %s", e)


def record_stream_quality(span, ttft_s: float | None, chunks: int, total_s: float,
                          output_tokens: int | None = None):
    """Attach streaming quality to the message span."""
    if span is None:
        return
    try:
        if ttft_s is not None:
            span.set_attribute("gen_ai.stream.ttft_s", round(ttft_s, 3))
        span.set_attribute("gen_ai.stream.chunks", chunks)
        span.set_attribute("gen_ai.stream.duration_s", round(total_s, 3))
        if total_s > 0:
            if output_tokens:
                span.set_attribute("gen_ai.stream.tokens_per_s",
                                   round(output_tokens / total_s, 2))
            else:
                span.set_attribute("gen_ai.stream.chunks_per_s",
                                   round(chunks / total_s, 2))
    except Exception as e:
        logger.debug("record_stream_quality failed: %s", e)


def record_cost(span, usage: dict, cost_usd: float | None):
    """Attach token counts and estimated cost to the message span."""
    if span is None or not usage:
        return
    try:
        for key, attr in (("model", "gen_ai.request.model"),
                          ("prompt_tokens", "gen_ai.usage.input_tokens"),
                          ("output_tokens", "gen_ai.usage.output_tokens"),
                          ("thinking_tokens", "gen_ai.usage.thinking_tokens"),
                          ("neurons", "gen_ai.usage.neurons")):
            value = usage.get(key)
            if value:
                span.set_attribute(attr, value)
        if cost_usd is not None:
            span.set_attribute("gen_ai.usage.cost_usd", round(cost_usd, 6))
    except Exception as e:
        logger.debug("record_cost failed: %s", e)


def flush():
    """Force-send buffered spans. Render can freeze the instance and drop the last trace."""
    if _llm_provider is None:
        return
    try:
        _llm_provider.force_flush()
    except Exception as e:
        logger.debug("flush failed: %s", e)
