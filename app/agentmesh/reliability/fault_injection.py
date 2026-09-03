"""Fault-injection harness — simulates a flaky downstream API.

Configurable per-tool failure rate. When enabled, tool implementations
have a configurable probability of raising an error, simulating a
saturated or flaky downstream service.

This is the chaos engineering primitive: inject failure before your
users find it for you.

Each agent creates its own FaultInjector instance so failure rates
never collide across agents sharing the platform's tool registry.
"""

import logging
import random


class InjectedFailureError(Exception):
    """Raised when a fault is injected to simulate a downstream failure."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(
            f"Injected failure for tool '{tool_name}' — simulating flaky downstream"
        )


class FaultInjector:
    """Per-agent fault injection harness.

    Usage:
        injector = FaultInjector("sourcing_agent")
        injector.set_failure_rate("query_suppliers", 0.40)
        injector.maybe_inject("query_suppliers")  # raises 40% of the time
    """

    def __init__(self, agent_name: str):
        self._logger = logging.getLogger(f"agentmesh.{agent_name}.fault_injection")
        self._failure_rates: dict[str, float] = {}

    def set_failure_rate(self, tool_name: str, rate: float) -> None:
        """Set the failure rate for a tool (0.0 to 1.0)."""
        if rate < 0.0 or rate > 1.0:
            raise ValueError(f"Failure rate must be 0.0-1.0, got {rate}")
        self._failure_rates[tool_name] = rate
        self._logger.info("FAULT_INJECTION_SET tool=%s rate=%.2f", tool_name, rate)

    def get_failure_rate(self, tool_name: str) -> float:
        """Get the current failure rate for a tool."""
        return self._failure_rates.get(tool_name, 0.0)

    def clear_failure_rates(self) -> None:
        """Clear all failure rates (disable fault injection)."""
        self._failure_rates.clear()
        self._logger.info("FAULT_INJECTION_CLEARED")

    def maybe_inject(self, tool_name: str) -> None:
        """Maybe inject a failure for the given tool.

        Raises InjectedFailureError with probability equal to the tool's
        configured failure rate. If no rate is set, this is a no-op.
        """
        rate = self._failure_rates.get(tool_name, 0.0)
        if rate > 0.0 and random.random() < rate:
            self._logger.warning(
                "FAULT_INJECTED tool=%s rate=%.2f — simulating downstream failure",
                tool_name,
                rate,
            )
            raise InjectedFailureError(tool_name)

    def is_enabled(self) -> bool:
        """Check if any fault injection is currently active."""
        return any(rate > 0.0 for rate in self._failure_rates.values())
