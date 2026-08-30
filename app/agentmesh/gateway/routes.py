import uuid
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
