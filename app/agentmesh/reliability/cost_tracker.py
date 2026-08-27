"""Cost tracker — accumulates cost within a single activity execution.

AgentMesh does NOT know model prices — that's InferRoute's job. AgentMesh
only tracks the budget. The cost_usd value comes from InferRoute's response
(computed by InferRoute using its registered pricing and actual token counts).

If InferRoute is not in the path (direct provider call), cost_usd is 0.0
and the tracker accumulates nothing — the budget check passes trivially.
"""

import logging
from dataclasses import dataclass, field

from app.agentmesh.llm.client import LLMResponse

logger = logging.getLogger("agentmesh.reliability.cost_tracker")


@dataclass
class CostTracker:
    """Accumulates cost within a single activity execution.

    Receives cost_usd from InferRoute's response (computed by InferRoute,
    not by AgentMesh). AgentMesh does NOT know model prices — that's
    InferRoute's job. AgentMesh only tracks the budget.
    """

    _spent_usd: float = 0.0
    _spent_tokens: int = 0
    _calls: list[dict] = field(default_factory=list)

    def record_llm_call(self, response: LLMResponse) -> None:
        """Record cost from an InferRoute LLMResponse."""
        self._spent_usd += response.cost_usd
        usage = response.usage or {}
        tokens = usage.get("total_tokens", 0)
        self._spent_tokens += tokens
        self._calls.append({
            "type": "llm",
            "model": response.model,
            "cost_usd": response.cost_usd,
            "tokens": tokens,
        })
        logger.debug(
            "COST_RECORDED type=llm model=%s cost=%.6f tokens=%d total=%.6f",
            response.model,
            response.cost_usd,
            tokens,
            self._spent_usd,
        )

    def record_tool_call(self, tool_name: str, cost_usd: float = 0.0) -> None:
        """Record cost from a tool call (tools typically cost $0)."""
        self._spent_usd += cost_usd
        self._calls.append({
            "type": "tool",
            "tool": tool_name,
            "cost_usd": cost_usd,
        })

    @property
    def total_usd(self) -> float:
        return self._spent_usd

    @property
    def total_tokens(self) -> int:
        return self._spent_tokens

    def is_over_budget(self, budget_usd: float) -> bool:
        """Check if accumulated cost exceeds the budget."""
        return self._spent_usd > budget_usd

    def summary(self) -> dict:
        """Return a summary dict for logging / SSE events / activity results."""
        return {
            "cost_incurred": round(self._spent_usd, 6),
            "tokens_used": self._spent_tokens,
            "call_count": len(self._calls),
            "calls": self._calls,
        }
