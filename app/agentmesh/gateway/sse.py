"""SSE (Server-Sent Events) formatting helpers.

Formats events for the text/event-stream media type:
    event: workflow_started
    id: abc123
    data: {"workflow_id": "sourcing-123", "timestamp": "..."}

"""

import json


def format_sse(event_type: str, data: dict, event_id: str | None = None) -> str:
    """Format a Server-Sent Event string.

    Args:
        event_type: The event type (e.g., "workflow_started").
        data: The event payload dict.
        event_id: Optional event ID (for reconnection via Last-Event-ID).

    Returns:
        A properly formatted SSE string ending with double newline.
    """
    lines = [f"event: {event_type}"]
    if event_id:
        lines.append(f"id: {event_id}")
    lines.append(f"data: {json.dumps(data)}")
    return "\n".join(lines) + "\n\n"
