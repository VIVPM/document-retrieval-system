"""OpenTelemetry tracing and metrics.

LLM spans export to Langfuse and Grafana off one provider; HTTP spans use a
separate provider so they don't also land in Langfuse. Each backend stays off
unless its env vars are set (LANGFUSE_* / GRAFANA_OTLP_*), and nothing here
raises — tracing must never break a request.
"""
import base64
import logging
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# This app doesn't configure root logging, and every init below is "log and
# continue" — so without a handler here a broken exporter (bad prod auth) would
# fail silently. Give the module its own stderr handler so those signals show.
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

_llm_provider = None   # unified TracerProvider for LLM spans, or None when disabled
_llm_tracer = None     # tracer from that provider (for the per-message parent span)
_message_counter = None
_cost_counter = None
_ttft_hist = None
_queue_depth_cb = None


def _have_langfuse() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _have_grafana() -> bool:
    return bool(os.getenv("GRAFANA_OTLP_ENDPOINT") and os.getenv("GRAFANA_OTLP_AUTH"))


def _resource():
    from opentelemetry.sdk.resources import Resource

    return Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME", "document-retrieval-system"),
        "service.namespace": "document-retrieval-system",
        "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development"),
    })


def init_observability():
    """Instrument google-genai; export LLM spans to whichever backends are configured."""
    global _llm_provider, _llm_tracer
    if not (_have_langfuse() or _have_grafana()):
        logger.info("LLM tracing disabled (no Langfuse/Grafana env).")
        return
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider = TracerProvider(resource=_resource())
        enabled = []

        if _have_langfuse():
            host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
            creds = f'{os.environ["LANGFUSE_PUBLIC_KEY"]}:{os.environ["LANGFUSE_SECRET_KEY"]}'
            auth = base64.b64encode(creds.encode()).decode()
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
                endpoint=f"{host}/api/public/otel/v1/traces",
                headers={"Authorization": f"Basic {auth}"},
            )))
            enabled.append(f"Langfuse ({host})")

        if _have_grafana():
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
                endpoint=f"{os.environ['GRAFANA_OTLP_ENDPOINT'].rstrip('/')}/v1/traces",
                headers={"Authorization": os.environ["GRAFANA_OTLP_AUTH"]},
            )))
            enabled.append("Grafana Cloud")

        # Instrument whichever SDK actually makes the calls. LLM_MODEL=CLOUDFLARE
        # routes generation through the openai client, which the google-genai
        # instrumentor never sees — without this the LLM traces would go silently
        # dark on that provider while HTTP spans kept flowing, which reads as a
        # working exporter. Embeddings are Gemini on both, so genai stays on.
        from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
        GoogleGenAIInstrumentor().instrument(tracer_provider=provider)
        if os.getenv("LLM_MODEL", "").strip().upper() == "CLOUDFLARE":
            from openinference.instrumentation.openai import OpenAIInstrumentor
            OpenAIInstrumentor().instrument(tracer_provider=provider)
            enabled.append("openai-sdk spans")
        _llm_provider = provider
        _llm_tracer = provider.get_tracer("chat")
        logger.info("LLM tracing enabled via OTLP: %s", ", ".join(enabled))
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
            # Name the trace explicitly: the FastAPI HTTP span (a different,
            # Grafana-only provider) sits in the active context as this span's
            # parent, and Langfuse never receives it — so without this the trace's
            # root name resolves empty. The nested genai spans still attach fine.
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
    """Attach streaming quality to the message span.

    TTFT and throughput are recorded SEPARATELY, and separately from total
    latency, because they move independently: a fast first token with a slow
    stream and a slow first token with a fast stream feel completely different
    to a user and are indistinguishable inside one total. Only the total was
    traced before.

    Throughput uses OUTPUT TOKENS when the provider reports them, and falls
    back to SSE chunks otherwise. The distinction matters more than it looks:
    providers chunk differently -- measured on the same prompt, Gemini sent 3
    chunks for 88 tokens and Cloudflare 46 for 101 -- so chunks/sec compares
    chunking strategy, not speed, and would make one provider look 25x faster
    than the other for no real reason.
    """
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
    """Attach token counts and estimated cost to the message span.

    Cost rather than tokens alone: rates differ per model and thinking tokens
    bill at the output rate, so a token count cannot answer "what did today
    cost" without a spreadsheet. A None cost sets no attribute at all, so an
    unpriced model shows as missing rather than as free.
    """
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


def init_http_tracing(app):
    """Trace every HTTP endpoint to Grafana, on its own provider so Langfuse stays LLM-only."""
    if not _have_grafana():
        logger.info("Grafana HTTP tracing disabled (GRAFANA_OTLP_* not set).")
        return
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        provider = TracerProvider(resource=_resource())
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
            endpoint=f"{os.environ['GRAFANA_OTLP_ENDPOINT'].rstrip('/')}/v1/traces",
            headers={"Authorization": os.environ["GRAFANA_OTLP_AUTH"]},
        )))
        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        logger.info("Grafana HTTP tracing enabled via OTLP.")
    except Exception:
        logger.exception("Grafana HTTP tracing init failed — continuing without it.")


def init_metrics():
    """Export a chat_messages_total counter. A metric, not traces, so alerts are plain PromQL."""
    global _message_counter, _cost_counter, _ttft_hist, _queue_depth_cb
    if not _have_grafana():
        return
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=f"{os.environ['GRAFANA_OTLP_ENDPOINT'].rstrip('/')}/v1/metrics",
                headers={"Authorization": os.environ["GRAFANA_OTLP_AUTH"]},
            ),
            export_interval_millis=15000,
        )
        provider = MeterProvider(resource=_resource(), metric_readers=[reader])
        meter = provider.get_meter("chat")
        _message_counter = meter.create_counter(
            "chat_messages_total",
            description="Chat messages handled, by status (ok/error)",
        )
        # Cost as a COUNTER, so a dashboard can rate() it into spend-per-hour
        # and a total. A gauge would only show the last answer's price, which
        # answers nothing about a day.
        _cost_counter = meter.create_counter(
            "llm_cost_usd_total",
            unit="USD",
            description="Estimated LLM spend, by model",
        )
        # A HISTOGRAM, not a counter or a gauge: TTFT is a latency distribution
        # and the tail is the part users complain about. A mean would hide it.
        _ttft_hist = meter.create_histogram(
            "llm_ttft_seconds",
            unit="s",
            description="Time to first token, by model",
        )

        # An OBSERVABLE gauge, not a counter: depth is a level, not an event.
        # The callback runs on the export interval, so the number is sampled
        # rather than pushed -- nothing has to remember to report it, and a
        # worker that dies stops contributing without leaving a stale value.
        def _observe_depth(_options):
            from opentelemetry.metrics import Observation
            try:
                import job_queue
                return [Observation(n, {"status": st})
                        for st, n in job_queue.depth().items()]
            except Exception:
                # A metrics callback must never raise: it runs on the exporter's
                # thread and would take the whole export down with it.
                return []

        _queue_depth_cb = meter.create_observable_gauge(
            "ingest_queue_depth",
            callbacks=[_observe_depth],
            description="Ingest jobs by status (queued/running/failed)",
        )
        logger.info("Grafana metrics enabled via OTLP.")
    except Exception:
        logger.exception("Grafana metrics init failed — continuing without it.")


def record_llm_metrics(usage: dict, cost_usd: float | None, ttft_s: float | None):
    """Feed cost and TTFT to Grafana.

    Separate from the span attributes: a span answers "what did THIS request
    do", a metric answers "what is happening overall". Alerting on spend or on
    a TTFT tail needs the second, and a trace backend is the wrong shape for it.
    """
    model = (usage or {}).get("model") or "unknown"
    try:
        if _cost_counter is not None and cost_usd:
            _cost_counter.add(cost_usd, {"model": model})
        if _ttft_hist is not None and ttft_s is not None:
            _ttft_hist.record(ttft_s, {"model": model})
    except Exception as e:
        logger.debug("record_llm_metrics failed: %s", e)


def record_message(status: str):
    """Increment the message counter."""
    if _message_counter is None:
        return
    try:
        _message_counter.add(1, {"status": status})
    except Exception as e:
        logger.debug("record_message failed: %s", e)
