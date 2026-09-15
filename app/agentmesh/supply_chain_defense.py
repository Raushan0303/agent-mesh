"""Supply chain + exfiltration + identity defenses (task_6).

Defends against three OWASP Agentic Applications Top 10 threats:
  - Sensitive information disclosure — tool outputs leak API keys, credit
    card numbers, or PII into the agent's context
  - Excessive agency — an agent accesses tools it shouldn't have
  - Supply chain — a tool dependency is compromised and exfiltrates data

Two defenses:
  1. Payload inspection (DLP) — scans tool outputs for sensitive data
     patterns (API keys, credit cards, SSNs, email addresses) before they
     enter the agent's context. Flagged content is redacted.
  2. Per-agent tool scoping — each agent gets an explicit allowlist of
     tools it can call. The registry enforces this — an agent that tries
     to call a tool outside its scope gets a ToolAccessDeniedError.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("agentmesh.supply_chain_defense")


class PayloadVerdict(str, Enum):
    CLEAN = "clean"
    FLAGGED = "flagged"
    REDACTED = "redacted"


@dataclass
class PayloadInspectionResult:
    verdict: PayloadVerdict = PayloadVerdict.CLEAN
    patterns_matched: list[str] = field(default_factory=list)
    original_payload: str = ""
    inspected_payload: str = ""


# Sensitive data patterns — DLP scanning
SENSITIVE_PATTERNS = [
    # API keys (common formats)
    (r"sk-[a-zA-Z0-9]{20,}", "api_key_openai"),
    (r"AKIA[A-Z0-9]{16}", "aws_access_key"),
    (r"ghp_[a-zA-Z0-9]{36}", "github_pat"),
    (r"xox[baprs]-[a-zA-Z0-9-]{10,}", "slack_token"),
    # Credit cards (basic pattern — not full Luhn validation)
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "credit_card_number"),
    # SSN (US format)
    (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),
    # Email addresses (in tool outputs — may indicate PII leakage)
    (r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "email_address"),
    # Bearer tokens
    (r"Bearer\s+[a-zA-Z0-9._-]+", "bearer_token"),
    # Private keys
    (r"-----BEGIN\s+(RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE\s+KEY-----", "private_key"),
]


def inspect_payload(payload: str | dict, redact: bool = True) -> PayloadInspectionResult:
    """Scan a tool output payload for sensitive data patterns.

    Args:
        payload: the tool output (string or dict — dict is JSON-serialized for scanning)
        redact: if True, matched patterns are replaced with [REDACTED:type]

    Returns:
        PayloadInspectionResult with verdict, matched patterns, and
        inspected payload (redacted if redact=True and patterns found).
    """
    if isinstance(payload, dict):
        text = str(payload)
    else:
        text = payload or ""

    if not text:
        return PayloadInspectionResult()

    matched = []
    inspected = text

    for pattern, name in SENSITIVE_PATTERNS:
        matches = list(re.finditer(pattern, text))
        if matches:
            matched.append(name)
            if redact:
                for m in matches:
                    inspected = inspected[:m.start()] + f"[REDACTED:{name}]" + inspected[m.end():]

    if matched:
        verdict = PayloadVerdict.REDACTED if redact else PayloadVerdict.FLAGGED
        logger.warning(
            "SENSITIVE_DATA_DETECTED patterns=%s redacted=%s",
            matched, redact,
        )
        return PayloadInspectionResult(
            verdict=verdict,
            patterns_matched=matched,
            original_payload=text,
            inspected_payload=inspected,
        )

    return PayloadInspectionResult(
        original_payload=text,
        inspected_payload=text,
    )


class ToolAccessDeniedError(Exception):
    """Raised when an agent tries to call a tool outside its scope."""
    pass


@dataclass
class ToolScope:
    """Per-agent tool scoping — which tools an agent is allowed to call.

    The registry enforces this: if an agent's scope doesn't include a tool,
    the call is denied before execution, regardless of whether the tool
    is registered.
    """
    agent_name: str
    allowed_tools: set[str] = field(default_factory=set)

    def can_call(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools

    def add_tool(self, tool_name: str) -> None:
        self.allowed_tools.add(tool_name)

    def remove_tool(self, tool_name: str) -> None:
        self.allowed_tools.discard(tool_name)


class ToolScopeRegistry:
    """Manages per-agent tool scopes. Agents can only call tools in their scope.

    This is the identity/privilege layer: each agent has an identity
    (its name) and a privilege set (its allowed tools). The registry
    enforces this at call time.
    """

    def __init__(self) -> None:
        self._scopes: dict[str, ToolScope] = {}

    def register_scope(self, agent_name: str, allowed_tools: list[str]) -> ToolScope:
        scope = ToolScope(agent_name=agent_name, allowed_tools=set(allowed_tools))
        self._scopes[agent_name] = scope
        logger.info(
            "TOOL_SCOPE_REGISTERED agent=%s tools=%s",
            agent_name, allowed_tools,
        )
        return scope

    def get_scope(self, agent_name: str) -> ToolScope | None:
        return self._scopes.get(agent_name)

    def check_access(self, agent_name: str, tool_name: str) -> bool:
        scope = self._scopes.get(agent_name)
        if scope is None:
            # No scope registered = no access (deny by default)
            logger.warning(
                "TOOL_ACCESS_DENIED agent=%s tool=%s — no scope registered",
                agent_name, tool_name,
            )
            return False
        if not scope.can_call(tool_name):
            logger.warning(
                "TOOL_ACCESS_DENIED agent=%s tool=%s — not in scope (allowed: %s)",
                agent_name, tool_name, scope.allowed_tools,
            )
            return False
        return True


# Singleton — agents register their scopes at import time
tool_scope_registry = ToolScopeRegistry()
