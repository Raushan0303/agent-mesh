import uuid
import time
import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from opentelemetry import trace

from app.agentmesh.gateway.event_emitter import get_event_emitter, get_redis
from app.agentmesh.gateway.registry import AGENT_REGISTRY
from app.agentmesh.gateway.sse import format_sse
from app.agentmesh.temporal.client import get_temporal_client
from app.agentmesh.observability.tracing import get_tracer

router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.post("")
async def start_workflow(request: Request) -> JSONResponse:
    tracer = get_tracer()
    data = await request.json()
    agent_type = data.get("agent_type")

    with tracer.start_as_current_span("gateway.start_workflow") as span:
        span.set_attribute("agent_type", agent_type)

        registration = AGENT_REGISTRY.get(agent_type)
        if not registration:
            span.set_attribute("error", True)
            return JSONResponse(
                {"error": f"Unknown agent_type: {agent_type}"},
                status_code=404,
            )

        try:
            validated_input = registration.input_model(**data.get("input", {}))
        except ValidationError as e:
            span.set_attribute("error", True)
            return JSONResponse({"error": e.errors()}, status_code=422)

        workflow_id = f"{agent_type}-{uuid.uuid4().hex[:8]}"
        span.set_attribute("workflow_id", workflow_id)

        # TracingInterceptor on the Temporal client automatically injects
        # the current OTel trace context into Temporal headers.
        # The worker's TracingInterceptor extracts it — no manual trace_id
        # passing needed. Domain models stay clean.
        client = await get_temporal_client()
        await client.start_workflow(
            registration.workflow_class.run,
            validated_input,
            id=workflow_id,
            task_queue=registration.task_queue,
        )

        # Extract trace_id from the current span for client correlation
        current_span = trace.get_current_span()
        span_context = current_span.get_span_context()
        trace_id = f"{span_context.trace_id:032x}" if span_context.is_valid else ""

        return JSONResponse({
            "workflow_id": workflow_id,
            "agent_type": agent_type,
            "trace_id": trace_id,
        })


@router.get("/{workflow_id}")
async def get_workflow_status(workflow_id: str) -> JSONResponse:
    client = await get_temporal_client()
    handle = client.get_workflow_handle(workflow_id)
    desc = await handle.describe()
    return JSONResponse(
        {
            "workflow_id": workflow_id,
            "status": desc.status.name,
            "run_id": desc.run_id,
        }
    )


@router.get("/{workflow_id}/stream")
async def stream_workflow(workflow_id: str, request: Request) -> StreamingResponse:
    """Stream workflow events via Server-Sent Events (SSE).

    Opens a long-lived HTTP connection that pushes workflow events as they
    happen. Events are published to Redis Pub/Sub by the Temporal worker
    and forwarded to the client here.

    Reconnection: if the client disconnects and reconnects with a
    `Last-Event-ID` header, missed events are replayed from Redis history
    before switching to the live stream.

    Event types:
      - workflow_started
      - activity_started / activity_completed
      - node_entered / node_exited
      - interrupt_fired
      - workflow_completed / workflow_failed
      - cost_exceeded
    """
    last_event_id = request.headers.get("Last-Event-ID")

    async def event_generator():
        try:
            emitter = await get_event_emitter()
            redis = await get_redis()

            # Replay missed events on reconnection
            if last_event_id:
                missed = await emitter.get_history(workflow_id, after_event_id=last_event_id)
                for event in missed:
                    yield format_sse(event["event_type"], event["data"], event["event_id"])

            # Send initial status
            try:
                client = await get_temporal_client()
                handle = client.get_workflow_handle(workflow_id)
                desc = await handle.describe()
                yield format_sse("workflow_status", {
                    "workflow_id": workflow_id,
                    "status": desc.status.name,
                })
            except Exception:
                yield format_sse("workflow_status", {
                    "workflow_id": workflow_id,
                    "status": "unknown",
                })

            # Subscribe to live events via Redis Pub/Sub
            pubsub = redis.pubsub()
            channel = f"workflow:{workflow_id}:events"
            await pubsub.subscribe(channel)

            try:
                # Use get_message with timeout to avoid blocking forever
                while True:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=30.0,
                    )
                    if message is not None and message["type"] == "message":
                        event = json.loads(message["data"])
                        yield format_sse(
                            event["event_type"],
                            event["data"],
                            event["event_id"],
                        )
                    else:
                        # Send a comment as keepalive on timeout
                        yield ": keepalive\n\n"
            finally:
                await pubsub.unsubscribe(channel)
                await pubsub.close()

        except asyncio.CancelledError:
            # Client disconnected — clean up
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


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
            "name": "activity.run_graph_until_interrupt",
            "duration_ms": 2800,
            "attributes": {
                "activity_type": "run_graph_until_interrupt",
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


@router.post("/{workflow_id}/signal")
async def send_signal(workflow_id: str, request: Request) -> JSONResponse:
    data = await request.json()
    signal_name = data.get("signal_name")
    payload = data.get("payload", {})

    client = await get_temporal_client()
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal(signal_name, payload)
    return JSONResponse({"workflow_id": workflow_id, "signal": signal_name})


@router.post("/{workflow_id}/approve")
async def approve_workflow(workflow_id: str, request: Request) -> JSONResponse:
    """Send an approval signal to a workflow paused at a human checkpoint.

    Agent-agnostic route: sends the "approve" signal with the human's
    decision. The workflow's signal handler receives it and resumes
    the graph via Command(resume=approval_data).

    Request body:
        {
            "approved": true,
            "comment": "Looks good, proceed"
        }
    """
    data = await request.json()
    approved = data.get("approved", False)
    comment = data.get("comment", "")

    approval_data = {"approved": approved, "comment": comment}

    client = await get_temporal_client()
    handle = client.get_workflow_handle(workflow_id)
    await handle.signal("approve", approval_data)
    return JSONResponse({
        "workflow_id": workflow_id,
        "signal": "approve",
        "approved": approved,
        "comment": comment,
    })


# ── Load test endpoints ──


@router.post("/benchmark")
async def run_benchmark(request: Request) -> JSONResponse:
    """Launch N concurrent benchmark workflows and collect results.

    Request body:
        {
            "count": 100,          # number of workflows to launch
            "sleep_ms": 100,       # activity sleep duration
            "payload_size": 100,   # dummy payload size
            "concurrency": 50,     # how many to launch in parallel
            "workers": 1           # how many workers to spin up for this test
        }

    Returns:
        {
            "total": 100,
            "completed": 98,
            "failed": 2,
            "throughput": 45.2,       # workflows/sec
            "p50_ms": 120,
            "p99_ms": 340,
            "duration_ms": 2210,
            "workers": 1
        }
    """
    data = await request.json()
    count = data.get("count", 100)
    sleep_ms = data.get("sleep_ms", 100)
    payload_size = data.get("payload_size", 100)
    concurrency = data.get("concurrency", 50)
    num_workers = data.get("workers", 1)

    client = await get_temporal_client()
    from app.agents.benchmark_agent.workflow import BenchmarkWorkflow
    from app.agents.benchmark_agent import TASK_QUEUE as BENCHMARK_TASK_QUEUE

    # Spawn additional benchmark workers as subprocesses
    # The main worker_runner already has 1 benchmark worker running.
    # We spawn (num_workers - 1) additional workers as separate processes.
    import subprocess
    import sys
    extra_procs = []
    for i in range(num_workers - 1):
        proc = subprocess.Popen(
            [sys.executable, "-c", """
import asyncio
from app.agentmesh.temporal.client import get_temporal_client
from app.agentmesh.temporal.worker import run_worker
from app.agents.benchmark_agent.workflow import BenchmarkWorkflow, benchmark_activity
from app.agents.benchmark_agent import TASK_QUEUE

async def main():
    client = await get_temporal_client()
    await run_worker(
        client=client,
        task_queue=TASK_QUEUE,
        workflows=[BenchmarkWorkflow],
        activities=[benchmark_activity],
        max_concurrent_activities=100,
    )

asyncio.run(main())
"""],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        extra_procs.append(proc)

    # Give extra workers a moment to start polling
    if extra_procs:
        await asyncio.sleep(2)

    # Launch workflows with bounded concurrency
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    latencies = []
    errors = []
    batch_id = uuid.uuid4().hex[:8]

    async def launch_one(i: int):
        async with semaphore:
            wf_id = f"benchmark-{batch_id}-{i:05d}"
            start = time.monotonic()
            try:
                handle = await client.start_workflow(
                    BenchmarkWorkflow.run,
                    {"sleep_ms": sleep_ms, "payload_size": payload_size},
                    id=wf_id,
                    task_queue=BENCHMARK_TASK_QUEUE,
                )
                result = await handle.result()
                elapsed_ms = (time.monotonic() - start) * 1000
                latencies.append(elapsed_ms)
                results.append({"id": wf_id, "status": "completed", "ms": round(elapsed_ms, 1)})
            except Exception as e:
                elapsed_ms = (time.monotonic() - start) * 1000
                errors.append({"id": wf_id, "error": str(e), "ms": round(elapsed_ms, 1)})

    overall_start = time.monotonic()
    await asyncio.gather(*[launch_one(i) for i in range(count)])
    overall_ms = (time.monotonic() - overall_start) * 1000

    # Kill extra worker subprocesses — they were only for this test
    for proc in extra_procs:
        proc.terminate()
    for proc in extra_procs:
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    # Calculate stats
    completed = len(results)
    failed = len(errors)
    throughput = round(completed / (overall_ms / 1000), 1) if overall_ms > 0 else 0

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) > 0 else 0
    avg = sum(latencies) / len(latencies) if latencies else 0

    return JSONResponse({
        "batch_id": batch_id,
        "total": count,
        "completed": completed,
        "failed": failed,
        "throughput_per_sec": throughput,
        "p50_ms": round(p50, 1),
        "p99_ms": round(p99, 1),
        "avg_ms": round(avg, 1),
        "duration_ms": round(overall_ms, 1),
        "concurrency": concurrency,
        "sleep_ms": sleep_ms,
        "workers": num_workers,
        "errors": errors[:5],  # first 5 errors if any
    })
