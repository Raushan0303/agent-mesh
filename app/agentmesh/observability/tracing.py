"""OpenTelemetry tracing — unified traces across Temporal + LangGraph + tools.

Stitches together:
  - Temporal's Workflow/Activity spans (via Temporal's OTel interceptor)
  - LangGraph's node-level execution spans (via our node wrapper)
  - Tool call spans (via the Tool Registry)

Into one trace per sourcing request, viewable end-to-end in Jaeger.

Agent-agnostic: instrumentation is at platform boundaries (gateway,
tool registry, memory store). Any agent's execution is traced automatically.
"""

import logging
from contextlib import asynccontextmanager
from functools import wraps

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.core.config import settings

logger = logging.getLogger("agentmesh.observability.tracing")

_tracer_provider: TracerProvider | None = None
_tracer: trace.Tracer | None = None


def setup_tracing(service_name: str = None) -> trace.Tracer:
    """Set up OpenTelemetry tracing with OTLP export to Jaeger.

    Returns a tracer that can be used to create spans.
    """
    global _tracer_provider, _tracer

    if _tracer is not None:
        return _tracer

    if not settings.otel_enabled:
        logger.info("OTEL_DISABLED — tracing is off")
        _tracer = trace.get_tracer(__name__)
        return _tracer

    resource = Resource.create(
        {"service.name": service_name or settings.otel_service_name}
    )

    _tracer_provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
    _tracer_provider.add_span_processor(
        BatchSpanProcessor(exporter)
    )

    trace.set_tracer_provider(_tracer_provider)
    _tracer = trace.get_tracer(__name__)

    logger.info(
        "OTEL_SETUP service=%s endpoint=%s",
        service_name or settings.otel_service_name,
        settings.otel_endpoint,
    )

    return _tracer


def get_tracer() -> trace.Tracer:
    """Get the global tracer (initializes if needed)."""
    global _tracer
    if _tracer is None:
        _tracer = setup_tracing()
    return _tracer


def traced_span(name: str, attributes: dict | None = None):
    """Decorator: wrap an async function in an OpenTelemetry span.

    Usage:
        @traced_span("query_suppliers")
        async def query_suppliers_impl(...): ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, str(value))
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.start_as_current_span(name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, str(value))
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@asynccontextmanager
async def trace_operation(name: str, attributes: dict | None = None):
    """Context manager: create a span for an async operation.

    Usage:
        async with trace_operation("hybrid_search", {"namespace": "sourcing"}):
            results = await hybrid_search(...)
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


def shutdown_tracing() -> None:
    """Shut down the tracer provider (flush pending spans)."""
    global _tracer_provider, _tracer
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        _tracer_provider = None
        _tracer = None
        logger.info("OTEL_SHUTDOWN")
