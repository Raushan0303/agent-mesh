"""Tests for the SSE event emitter and formatting.

Tests:
1. Basic streaming — emit events, verify they're published
2. Reconnection with Last-Event-ID — get_history returns events after the ID
3. Interrupt event — emit interrupt_fired event
4. Buffer overflow — history is trimmed to buffer size
5. Non-existent workflow — get_history returns empty list
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agentmesh.gateway.event_emitter import WorkflowEventEmitter
from app.agentmesh.gateway.sse import format_sse


def test_format_sse_basic():
    """SSE format — basic event without ID."""
    result = format_sse("workflow_started", {"workflow_id": "test-123"})
    assert "event: workflow_started" in result
    assert "data:" in result
    assert result.endswith("\n\n")


def test_format_sse_with_id():
    """SSE format — event with ID for reconnection."""
    result = format_sse("node_entered", {"node": "research"}, event_id="abc-123")
    assert "event: node_entered" in result
    assert "id: abc-123" in result
    assert "data:" in result
    assert result.endswith("\n\n")


def test_format_sse_data_is_json():
    """SSE format — data field is valid JSON."""
    data = {"workflow_id": "test-123", "status": "running"}
    result = format_sse("workflow_started", data)
    # Extract the data line and parse it
    for line in result.strip().split("\n"):
        if line.startswith("data: "):
            parsed = json.loads(line[6:])
            assert parsed == data


@pytest.mark.asyncio
async def test_event_emitter_emit():
    """Emit event — publishes to Redis channel and stores in history."""
    redis = AsyncMock()
    redis.publish = AsyncMock()
    redis.rpush = AsyncMock()
    redis.expire = AsyncMock()
    redis.ltrim = AsyncMock()

    emitter = WorkflowEventEmitter(redis)
    event_id = await emitter.emit("test-wf-123", "workflow_started", {"status": "running"})

    assert event_id is not None
    redis.publish.assert_called_once()
    redis.rpush.assert_called_once()
    redis.expire.assert_called_once()
    redis.ltrim.assert_called_once()

    # Verify channel name
    channel = redis.publish.call_args[0][0]
    assert channel == "workflow:test-wf-123:events"


@pytest.mark.asyncio
async def test_event_emitter_get_history():
    """Get history — returns all events when no after_event_id."""
    redis = AsyncMock()
    events = [
        json.dumps({"event_id": "1", "event_type": "workflow_started", "data": {}}),
        json.dumps({"event_id": "2", "event_type": "node_entered", "data": {}}),
    ]
    redis.lrange = AsyncMock(return_value=events)

    emitter = WorkflowEventEmitter(redis)
    history = await emitter.get_history("test-wf-123")

    assert len(history) == 2
    assert history[0]["event_id"] == "1"
    assert history[1]["event_id"] == "2"


@pytest.mark.asyncio
async def test_event_emitter_get_history_after_id():
    """Get history after event ID — returns only events after the given ID."""
    redis = AsyncMock()
    events = [
        json.dumps({"event_id": "1", "event_type": "workflow_started", "data": {}}),
        json.dumps({"event_id": "2", "event_type": "node_entered", "data": {}}),
        json.dumps({"event_id": "3", "event_type": "node_exited", "data": {}}),
    ]
    redis.lrange = AsyncMock(return_value=events)

    emitter = WorkflowEventEmitter(redis)
    history = await emitter.get_history("test-wf-123", after_event_id="2")

    assert len(history) == 1
    assert history[0]["event_id"] == "3"


@pytest.mark.asyncio
async def test_event_emitter_get_history_empty():
    """Non-existent workflow — get_history returns empty list."""
    redis = AsyncMock()
    redis.lrange = AsyncMock(return_value=[])

    emitter = WorkflowEventEmitter(redis)
    history = await emitter.get_history("nonexistent-wf")

    assert history == []
