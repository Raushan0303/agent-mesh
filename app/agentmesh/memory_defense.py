"""Memory + context defenses (task_5).

Defends against two OWASP Agentic Applications Top 10 threats:
  - Memory poisoning — untrusted data (tool outputs, user input) overwrites
    trusted agent decisions in the memory store
  - Context privilege escalation — untrusted context is injected into the
    agent's reasoning with the same privilege level as system instructions

Three defenses:
  1. Origin tagging — every memory entry is tagged with its origin
     (user_input, tool_output, agent_decision, feedback) so the agent
     knows where each piece of context came from
  2. Privilege tagging — memory entries get a privilege level (trusted,
     untrusted, verified) so the agent can weigh context by source
  3. Integrity hash — memory entries are hashed to detect tampering

The retrieval layer can then filter by privilege level — e.g. "only
retrieve from trusted or verified sources when making side-effecting
decisions."
"""

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("agentmesh.memory_defense")


class MemoryOrigin(str, Enum):
    """Where a memory entry came from."""
    SYSTEM = "system"              # system prompt, hardcoded rules
    USER_INPUT = "user_input"      # user-supplied brief fields
    TOOL_OUTPUT = "tool_output"    # results from tool calls (untrusted)
    AGENT_DECISION = "agent_decision"  # LLM decisions (semi-trusted)
    HUMAN_FEEDBACK = "human_feedback"  # human approval/rejection (trusted)
    VERIFIED = "verified"          # independently verified (trusted)


class PrivilegeLevel(str, Enum):
    """How much the agent should trust a memory entry."""
    TRUSTED = "trusted"        # system, human_feedback, verified
    SEMI_TRUSTED = "semi_trusted"  # agent_decision
    UNTRUSTED = "untrusted"    # user_input, tool_output


ORIGIN_PRIVILEGE_MAP = {
    MemoryOrigin.SYSTEM: PrivilegeLevel.TRUSTED,
    MemoryOrigin.HUMAN_FEEDBACK: PrivilegeLevel.TRUSTED,
    MemoryOrigin.VERIFIED: PrivilegeLevel.TRUSTED,
    MemoryOrigin.AGENT_DECISION: PrivilegeLevel.SEMI_TRUSTED,
    MemoryOrigin.USER_INPUT: PrivilegeLevel.UNTRUSTED,
    MemoryOrigin.TOOL_OUTPUT: PrivilegeLevel.UNTRUSTED,
}


@dataclass
class TaggedMemoryEntry:
    """A memory entry with origin, privilege, and integrity metadata."""
    content: str
    origin: MemoryOrigin
    privilege: PrivilegeLevel = field(init=False)
    integrity_hash: str = field(init=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.privilege = ORIGIN_PRIVILEGE_MAP[self.origin]
        self.integrity_hash = _hash_entry(self.content, self.origin.value)

    def verify_integrity(self) -> bool:
        """Check if the entry's content matches its integrity hash."""
        return _hash_entry(self.content, self.origin.value) == self.integrity_hash

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "origin": self.origin.value,
            "privilege": self.privilege.value,
            "integrity_hash": self.integrity_hash,
            "metadata": self.metadata,
        }


def _hash_entry(content: str, origin: str) -> str:
    """Compute a SHA256 hash of the content + origin for integrity checking."""
    return hashlib.sha256(f"{origin}:{content}".encode()).hexdigest()[:16]


def tag_memory_entry(content: str, origin: MemoryOrigin, **metadata) -> TaggedMemoryEntry:
    """Create a tagged memory entry with automatic privilege assignment."""
    return TaggedMemoryEntry(content=content, origin=origin, metadata=metadata)


def filter_by_privilege(
    entries: list[TaggedMemoryEntry],
    min_privilege: PrivilegeLevel = PrivilegeLevel.SEMI_TRUSTED,
) -> list[TaggedMemoryEntry]:
    """Filter memory entries by minimum privilege level.

    Use TRUSTED for side-effecting decisions (only system + human feedback + verified).
    Use SEMI_TRUSTED for reasoning (adds agent decisions).
    Use UNTRUSTED for all context (includes user input + tool output).
    """
    privilege_order = {
        PrivilegeLevel.UNTRUSTED: 0,
        PrivilegeLevel.SEMI_TRUSTED: 1,
        PrivilegeLevel.TRUSTED: 2,
    }
    min_level = privilege_order[min_privilege]
    return [e for e in entries if privilege_order[e.privilege] >= min_level]


def detect_poisoning(
    entries: list[TaggedMemoryEntry],
    trusted_origins: set[MemoryOrigin] | None = None,
) -> list[TaggedMemoryEntry]:
    """Detect potential memory poisoning — untrusted entries that look like
    trusted instructions.

    Flags entries from untrusted origins (user_input, tool_output) that
    contain instruction-like patterns (e.g. "you must", "always do",
    "never do"). These could be attempts to poison the agent's memory
    with fake trusted instructions.
    """
    if trusted_origins is None:
        trusted_origins = {MemoryOrigin.SYSTEM, MemoryOrigin.HUMAN_FEEDBACK, MemoryOrigin.VERIFIED}

    instruction_patterns = [
        "you must", "you should", "always do", "never do",
        "you are", "your role", "your task", "your instructions",
    ]

    poisoned = []
    for entry in entries:
        if entry.origin in trusted_origins:
            continue
        content_lower = entry.content.lower()
        for pattern in instruction_patterns:
            if pattern in content_lower:
                logger.warning(
                    "MEMORY_POISONING_DETECTED origin=%s pattern='%s' content_preview=%.60s",
                    entry.origin.value, pattern, entry.content[:60],
                )
                poisoned.append(entry)
                break

    return poisoned
