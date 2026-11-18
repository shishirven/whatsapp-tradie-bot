"""OpenTelemetry tracing. Off unless an OTLP endpoint is configured.

One trace per customer message:

    whatsapp.turn                       turn id, conversation, job, outcome
      └─ chat <model>                   the qualify call (GenAI semantic conventions)

Set these to send traces to any OTLP backend (Langfuse, Arize Phoenix, Grafana,
Honeycomb, AWS X-Ray via the ADOT collector, ...):

    OTEL_EXPORTER_OTLP_ENDPOINT   e.g. https://cloud.langfuse.com/api/public/otel
    OTEL_EXPORTER_OTLP_HEADERS    e.g. Authorization=Basic <base64 public:secret>
    OTEL_SERVICE_NAME             optional, defaults to whatsapp-tradie-bot

Spans carry ids, models, timings, token counts, hazards and guardrail rule
names. They do not carry phone numbers or message text; that stays in the
app's own database (`llm_calls` and `messages`).

If the OpenTelemetry packages are missing or the endpoint is unset, every
call here is a no-op and the app behaves exactly as before.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("tradie.telemetry")

try:  # pragma: no cover - exercised implicitly
    from opentelemetry import trace
except ImportError:  # pragma: no cover
    trace = None  # type: ignore[assignment]

_configured = False


def setup_tracing() -> bool:
    """Install an OTLP exporter once, if an endpoint is set. Returns True when traces are exported."""
    global _configured
    if _configured:
        return True
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint or trace is None:
        return False
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT is set but the OpenTelemetry SDK is not installed")
        return False
    service = os.getenv("OTEL_SERVICE_NAME") or "whatsapp-tradie-bot"
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    # The exporter reads the endpoint and headers from the standard OTEL_* variables.
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True
    logger.info("OpenTelemetry traces export to %s as %s", _host_only(endpoint), service)
    return True


def setup_logging() -> None:
    """Send the app's own `tradie.*` logs to stdout at LOG_LEVEL (default INFO). Idempotent.

    Without this, Python drops every INFO line from the app (uvicorn only sets up its
    own loggers), so lines such as "gemini qualify answered with model ..." never appear.
    """
    app_logger = logging.getLogger("tradie")
    if any(getattr(handler, "_tradie", False) for handler in app_logger.handlers):
        return
    handler = logging.StreamHandler()
    handler._tradie = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    app_logger.addHandler(handler)
    level = (os.getenv("LOG_LEVEL") or "INFO").upper()
    app_logger.setLevel(getattr(logging, level, logging.INFO))


def tracer():
    if trace is None:
        return None
    return trace.get_tracer("tradie")


@contextmanager
def span(name: str, attributes: dict | None = None) -> Iterator[object]:
    """A span when tracing is available (a no-op span otherwise)."""
    current = tracer()
    if current is None:
        yield None
        return
    with current.start_as_current_span(name, attributes=_clean(attributes or {})) as active:
        yield active


def set_attributes(target, attributes: dict) -> None:
    """Set attributes on a span; None values are skipped. Never raises."""
    if target is None:
        return
    try:
        target.set_attributes(_clean(attributes))
    except Exception:  # pragma: no cover
        logger.debug("could not set span attributes", exc_info=True)


def current_span():
    if trace is None:
        return None
    return trace.get_current_span()


def model_call_attributes(meta, *, operation: str = "chat") -> dict:
    """GenAI semantic-convention attributes for one model call."""
    provider = getattr(meta, "provider", None)
    attempts = getattr(meta, "attempts", None) or []
    requested = attempts[0].get("model") if attempts else None
    return {
        "gen_ai.operation.name": operation,
        "gen_ai.provider.name": {"gemini": "gcp.gemini", "anthropic": "anthropic"}.get(provider, provider),
        "gen_ai.request.model": requested,
        "gen_ai.response.model": getattr(meta, "model", None),
        "gen_ai.usage.input_tokens": getattr(meta, "input_tokens", None),
        "gen_ai.usage.output_tokens": getattr(meta, "output_tokens", None),
        "tradie.llm.status": getattr(meta, "status", None),
        "tradie.llm.fallback_from": getattr(meta, "fallback_from", None),
        "tradie.llm.attempts": len(attempts) or None,
    }


def _clean(attributes: dict) -> dict:
    cleaned = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def _host_only(endpoint: str) -> str:
    """Log the endpoint's host, never a path or query that could hold a key."""
    without_scheme = endpoint.split("://", 1)[-1]
    return without_scheme.split("/", 1)[0]
