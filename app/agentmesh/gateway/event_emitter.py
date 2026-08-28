"""Workflow event emitter — publishes events to Redis Pub/Sub.

Called by the Temporal worker (activity wrapper) and LangGraph (node wrapper).
Events are published to channel:
    workflow:{workflow_id}:events

The gateway's SSE endpoint subscribes to this channel and forwards events
to the connected client. Event history is stored in a Redis list for
reconnection replay (when a client disconnects and reconnects with
Last-Event-ID, the gateway replays missed events from history).
"""

import json
import logging
import uuid
from datetime import datetime, timezone

from app.core.config import settings
from app.core.constants import SSE_CHANNEL_PREFIX, SSE_HISTORY_SUFFIX

logger = logging.getLogger("agentmesh.gateway.event_emitter")


class WorkflowEventEmitter:
    """Publishes workflow events to Redis Pub/Sub.

    Called by the Temporal worker (activity wrapper) and LangGraph
    (node wrapper). Events are published to channel:
        workflow:{workflow_id}:events

    The gateway's SSE endpoint subscribes to this channel and
    forwards events to the connected client.
    """

    def __init__(self, redis):
        self._redis = redis

    @staticmethod
    def _channel_name(workflow_id: str) -> str:
        return f"{SSE_CHANNEL_PREFIX}:{workflow_id}:events"

    @staticmethod
    def _history_key(workflow_id: str) -> str:
        return f"{SSE_CHANNEL_PREFIX}:{workflow_id}:{SSE_HISTORY_SUFFIX}"

    async def emit(self, workflow_id: str, event_type: str, data: dict) -> str:
        """Publish an event to Redis Pub/Sub.

        Args:
            workflow_id: The workflow ID this event belongs to.
            event_type: Event type (e.g., "workflow_started", "node_entered").
            data: Event payload dict.

        Returns:
            The event ID (UUID) of the published event.
        """
        event_id = str(uuid.uuid4())
        event = {
            "event_id": event_id,
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        serialized = json.dumps(event)

        channel = self._channel_name(workflow_id)
        history_key = self._history_key(workflow_id)

        # Publish to channel (live subscribers receive this)
        await self._redis.publish(channel, serialized)

        # Store in history list (for reconnection replay)
        await self._redis.rpush(history_key, serialized)
        await self._redis.expire(history_key, settings.sse_history_ttl_seconds)

        # Trim history to max buffer size
        await self._redis.ltrim(history_key, -settings.sse_buffer_size, -1)

        logger.debug(
            "EVENT_EMIT workflow_id=%s type=%s event_id=%s",
            workflow_id, event_type, event_id,
        )
        return event_id

    async def get_history(self, workflow_id: str, after_event_id: str | None = None) -> list[dict]:
        """Get event history for reconnection replay.

        Args:
            workflow_id: The workflow ID.
            after_event_id: If provided, only return events after this event ID.
                If None, return all history.

        Returns:
            List of event dicts (in order).
        """
        history_key = self._history_key(workflow_id)
        raw_events = await self._redis.lrange(history_key, 0, -1)

        events = []
        found_after = after_event_id is None
        for raw in raw_events:
            event = json.loads(raw)
            if after_event_id is not None:
                if found_after:
                    events.append(event)
                elif event["event_id"] == after_event_id:
                    found_after = True
            else:
                events.append(event)

        return events


# ── Singleton management ──

_emitter: WorkflowEventEmitter | None = None
_redis = None


async def get_redis():
    """Get or create the Redis connection singleton."""
    global _redis
    if _redis is not None:
        return _redis
    import redis.asyncio as aioredis
    _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def get_event_emitter() -> WorkflowEventEmitter:
    """Get or create the WorkflowEventEmitter singleton."""
    global _emitter
    if _emitter is not None:
        return _emitter
    redis = await get_redis()
    _emitter = WorkflowEventEmitter(redis)
    return _emitter


def reset_event_emitter() -> None:
    """Reset the singleton (for tests)."""
    global _emitter, _redis
    _emitter = None
    _redis = None
