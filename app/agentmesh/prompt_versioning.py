"""Prompt versioning — hash and tag every prompt sent to the LLM.

Every LLM call in AgentMesh records a prompt_hash (SHA256 of the rendered
prompt) and prompt_version (human-readable tag) in the graph state. This
gets checkpointed in Postgres by LangGraph, so every past run has a
permanent record of which prompt was used.

Usage:
    from app.agentmesh.prompt_versioning import version_prompt

    messages = build_decide_prompt(item, quantity, ...)
    tagged = version_prompt(messages, version="v1")

    # tagged.messages — the LLM messages (pass to client.complete)
    # tagged.hash    — SHA256[:16] of the rendered messages
    # tagged.version — "v1" (human-readable, bumped manually)

When you change a prompt, bump the version string. The hash changes
automatically. Old runs keep their hash + version in the checkpoint.
New runs get the new ones.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptVersion:
    """Immutable record of which prompt was sent to the LLM.

    Attributes:
        messages: The LLM messages (list of LLMMessage dicts with role + content).
        hash: SHA256[:16] of the serialized messages. Deterministic — same
            messages always produce the same hash.
        version: Human-readable version tag (e.g. "v1", "v2"). Set manually
            in the prompt builder. Bump it when you change the prompt.
    """

    messages: list[dict[str, str]]
    hash: str
    version: str


def hash_prompt(messages: list[dict[str, str]]) -> str:
    """Compute a deterministic hash of the rendered prompt.

    The hash is SHA256 of the JSON-serialized messages, sorted by key
    to ensure ordering independence. Truncated to 16 chars for readability.

    Two prompts with the same messages always produce the same hash.
    Changing a single character in any message changes the hash.
    """
    # Normalize: convert LLMMessage dataclass instances to dicts if needed
    normalized = []
    for msg in messages:
        if hasattr(msg, "role") and hasattr(msg, "content"):
            normalized.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg, dict):
            normalized.append({"role": msg["role"], "content": msg["content"]})
        else:
            raise TypeError(f"Unexpected message type: {type(msg)}")

    # Sort keys for ordering independence, encode as UTF-8
    serialized = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def version_prompt(
    messages: list[Any],
    version: str,
) -> PromptVersion:
    """Tag a prompt with its hash and version.

    Args:
        messages: The LLM messages (list of LLMMessage or dict).
        version: Human-readable version tag (e.g. "v1").

    Returns:
        PromptVersion with the messages, hash, and version.
    """
    return PromptVersion(
        messages=messages,
        hash=hash_prompt(messages),
        version=version,
    )
