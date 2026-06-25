from temporalio.client import Client

from app.core.config import settings

_client: Client | None = None


async def get_temporal_client() -> Client:
    global _client
    if _client is None:
        # TracingInterceptor propagates W3C trace context through Temporal
        # headers — the gateway's trace ID flows to the worker automatically.
        # No need to pass trace_id as workflow input data.
        from temporalio.contrib.opentelemetry import TracingInterceptor
        from app.agentmesh.observability.tracing import get_tracer

        _client = await Client.connect(
            settings.temporal_host,
            namespace=settings.temporal_namespace,
            interceptors=[TracingInterceptor(tracer=get_tracer())],
        )
    return _client
