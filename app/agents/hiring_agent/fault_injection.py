"""Fault-injection harness for the hiring agent's tools — same pattern as
sourcing_agent/fault_injection.py, scoped to this agent's own tool names
(schedule_interview, send_offer) so failure rates never collide across
agents sharing the platform's tool registry.
"""

import logging
import random

logger = logging.getLogger("agentmesh.hiring_agent.fault_injection")

_failure_rates: dict[str, float] = {}


class InjectedFailureError(Exception):
    """Raised when a fault is injected to simulate a downstream failure."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(
            f"Injected failure for tool '{tool_name}' — simulating flaky downstream"
        )


def set_failure_rate(tool_name: str, rate: float) -> None:
    if rate < 0.0 or rate > 1.0:
        raise ValueError(f"Failure rate must be 0.0-1.0, got {rate}")
    _failure_rates[tool_name] = rate
    logger.info("FAULT_INJECTION_SET tool=%s rate=%.2f", tool_name, rate)


def get_failure_rate(tool_name: str) -> float:
    return _failure_rates.get(tool_name, 0.0)


def clear_failure_rates() -> None:
    _failure_rates.clear()
    logger.info("FAULT_INJECTION_CLEARED")


def maybe_inject(tool_name: str) -> None:
    rate = _failure_rates.get(tool_name, 0.0)
    if rate > 0.0 and random.random() < rate:
        logger.warning(
            "FAULT_INJECTED tool=%s rate=%.2f — simulating downstream failure",
            tool_name,
            rate,
        )
        raise InjectedFailureError(tool_name)


def is_fault_injection_enabled() -> bool:
    return any(rate > 0.0 for rate in _failure_rates.values())
