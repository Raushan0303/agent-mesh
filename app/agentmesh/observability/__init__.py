"""Observability — OpenTelemetry distributed tracing.

Stitches together Temporal's Workflow/Activity spans with LangGraph's
node-level execution and tool calls, into one trace per request.
"""

from app.agentmesh.observability.tracing import (
    get_tracer,
    setup_tracing,
    shutdown_tracing,
    trace_operation,
    traced_span,
)

__all__ = [
    "get_tracer",
    "setup_tracing",
    "shutdown_tracing",
    "trace_operation",
    "traced_span",
]
