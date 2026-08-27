"""Output size cap — truncates large tool results before they reach the LLM.

After a tool executes, before the result is returned to the LLM, the sandbox
checks the size of the result. If it exceeds max_output_bytes, the result is
truncated and a _truncated: true field is added. Default 64KB, configurable
per-tool via ToolSpec.max_output_bytes.

This prevents a database query tool returning 100,000 rows (a 20MB JSON blob)
from blowing the LLM's context window. The LLM sees the first chunk and a
_truncated: true flag indicating there's more data.
"""

import json
import logging

logger = logging.getLogger("agentmesh.tool_registry.output_cap")


def enforce_output_cap(result: dict, max_bytes: int) -> dict:
    """Enforce a maximum output size on a tool result.

    If the serialized result exceeds max_bytes:
    - Truncate large string/list fields to fit within the budget
    - Add metadata about the truncation
    - Log a warning

    Returns the (possibly truncated) result dict.
    """
    serialized = json.dumps(result)
    size = len(serialized.encode("utf-8"))

    if size <= max_bytes:
        return result

    logger.warning(
        "OUTPUT_CAPPED size=%d max=%d — truncating",
        size,
        max_bytes,
    )

    truncated = _truncate_fields(result, max_bytes)
    truncated["_truncated"] = True
    truncated["_original_size_bytes"] = size
    truncated["_max_bytes"] = max_bytes
    return truncated


def _truncate_fields(result: dict, max_bytes: int) -> dict:
    """Truncate the largest fields until under the byte budget."""
    result = dict(result)  # shallow copy

    # Sort fields by serialized size, largest first
    fields_by_size = sorted(
        result.keys(),
        key=lambda k: len(json.dumps(result[k]).encode("utf-8")) if not isinstance(result[k], str) else len(result[k].encode("utf-8")),
        reverse=True,
    )

    for field in fields_by_size:
        if len(json.dumps(result).encode("utf-8")) <= max_bytes:
            break
        value = result[field]
        if isinstance(value, str):
            result[field] = value[: max_bytes // 4] + "...[truncated]"
        elif isinstance(value, list):
            result[field] = _truncate_list(value, max_bytes // 2)
        elif isinstance(value, dict):
            result[field] = _truncate_fields(value, max_bytes // 2)
        # Other types (int, float, bool) are small — skip

    return result


def _truncate_list(items: list, max_bytes: int) -> list:
    """Keep the first N items that fit within the byte budget."""
    truncated = []
    current_size = 2  # "[]"
    for item in items:
        item_size = len(json.dumps(item).encode("utf-8")) + 1  # +1 for comma
        if current_size + item_size > max_bytes:
            break
        truncated.append(item)
        current_size += item_size

    if len(truncated) < len(items):
        truncated.append(f"...[{len(items) - len(truncated)} more items truncated]")

    return truncated
