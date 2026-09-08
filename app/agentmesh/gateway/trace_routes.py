"""Trace reconstruction route — builds a simulated trace tree for the UI."""

import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.agentmesh.temporal.client import get_temporal_client

router = APIRouter(prefix="/workflows", tags=["tracing"])


@router.get("/{workflow_id}/trace")
async def get_workflow_trace(workflow_id: str) -> JSONResponse:
    """Return a simulated trace tree for a workflow.

    This reconstructs the trace from the workflow's known structure.
    In production, you'd query Jaeger's API with the trace_id.
    Here we build the trace tree from the workflow definition so the
    UI can show what the distributed trace looks like.
    """
    client = await get_temporal_client()
    handle = client.get_workflow_handle(workflow_id)
    desc = await handle.describe()

    # Build the trace tree from the known workflow structure
    # In a real system, these spans would be exported to Jaeger via OTLP
    # and you'd query Jaeger's API. Here we reconstruct them.
    spans = [
        {
            "span_id": "root",
            "parent_id": None,
            "service": "agentmesh-gateway",
            "name": "gateway.start_workflow",
            "duration_ms": 15,
            "attributes": {
                "agent_type": "sourcing_agent",
                "workflow_id": workflow_id,
                "http.method": "POST",
                "http.url": "/workflows",
            },
        },
        {
            "span_id": "wf-run",
            "parent_id": "root",
            "service": "agentmesh-worker",
            "name": "workflow.run (Temporal)",
            "duration_ms": 3200,
            "attributes": {
                "workflow_type": "SourcingWorkflow",
                "task_queue": "sourcing-task-queue",
                "workflow_status": desc.status.name,
            },
        },
        {
            "span_id": "act-1",
            "parent_id": "wf-run",
            "service": "agentmesh-worker",
            "name": "activity.run_sourcing_graph",
            "duration_ms": 2800,
            "attributes": {
                "activity_type": "run_sourcing_graph",
                "retry_policy": "aggressive",
            },
        },
        {
            "span_id": "node-research",
            "parent_id": "act-1",
            "service": "agentmesh-worker",
            "name": "langgraph.node.research",
            "duration_ms": 850,
            "attributes": {
                "node": "research",
                "tool": "query_suppliers",
                "suppliers_found": 5,
            },
        },
        {
            "span_id": "node-score",
            "parent_id": "act-1",
            "service": "agentmesh-worker",
            "name": "langgraph.node.score",
            "duration_ms": 120,
            "attributes": {
                "node": "score",
                "scoring_criteria": "price,rating,lead_time",
            },
        },
        {
            "span_id": "node-decide",
            "parent_id": "act-1",
            "service": "agentmesh-worker",
            "name": "langgraph.node.decide",
            "duration_ms": 1800,
            "attributes": {
                "node": "decide",
                "llm_provider": "inferroute",
                "past_decisions_queried": 2,
            },
        },
        {
            "span_id": "inferroute-call",
            "parent_id": "node-decide",
            "service": "inferroute-gateway",
            "name": "POST /v1/chat/completions",
            "duration_ms": 1700,
            "attributes": {
                "service": "infer-route",
                "w3c.traceparent": "propagated from agentmesh-worker",
            },
        },
        {
            "span_id": "ir-classify",
            "parent_id": "inferroute-call",
            "service": "inferroute-gateway",
            "name": "routing.classify_complexity",
            "duration_ms": 2,
            "attributes": {
                "complexity": "medium",
                "tier": "standard",
                "strategy": "intelligence_aware",
                "source": "heuristic",
            },
        },
        {
            "span_id": "ir-cache",
            "parent_id": "inferroute-call",
            "service": "inferroute-gateway",
            "name": "cache.lookup",
            "duration_ms": 4,
            "attributes": {
                "exact_cache": "miss",
                "semantic_cache": "miss",
                "coalescer": "n/a",
            },
        },
        {
            "span_id": "ir-provider",
            "parent_id": "inferroute-call",
            "service": "inferroute-gateway",
            "name": "provider.call",
            "duration_ms": 1680,
            "attributes": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "input_tokens": 342,
                "output_tokens": 180,
                "cost": "$0.000051",
                "cache_hit": False,
            },
        },
        {
            "span_id": "node-approve",
            "parent_id": "wf-run",
            "service": "agentmesh-worker",
            "name": "langgraph.node.approve (interrupt)",
            "duration_ms": 0,
            "attributes": {
                "node": "approve",
                "checkpoint": "interrupt()",
                "waiting_for": "human signal",
                "zero_worker_cost": True,
            },
        },
        {
            "span_id": "node-confirm",
            "parent_id": "wf-run",
            "service": "agentmesh-worker",
            "name": "langgraph.node.confirm",
            "duration_ms": 350,
            "attributes": {
                "node": "confirm",
                "po_created": True,
                "payment_initiated": True,
            },
        },
    ]

    return JSONResponse({
        "workflow_id": workflow_id,
        "status": desc.status.name,
        "trace_id": f"{uuid.uuid4().hex[:32]}",
        "spans": spans,
        "services": ["agentmesh-gateway", "agentmesh-worker", "inferroute-gateway"],
        "jaeger_url": "http://localhost:16686",
    })
