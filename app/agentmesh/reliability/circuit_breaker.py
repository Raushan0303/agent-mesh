"""Circuit breaker — per-tool failure protection across Workflows.

Temporal's Activity retries are per-Workflow: if Activity A fails in
Workflow 1, Temporal retries it in Workflow 1. But it doesn't tell
Workflow 2 that the downstream is saturated. The circuit breaker is
the cross-Workflow bulkhead Temporal doesn't give you.

States:
  CLOSED    — calls pass through, failures are counted
  OPEN      — calls fail fast (no downstream call), after N consecutive failures
  HALF_OPEN — limited calls allowed to probe recovery

Keyed by tool name, not by agent — the same CircuitBreaker class protects
any agent's tools today and any future agent's tools tomorrow, each with
independent state.
"""

import asyncio
import logging
import time
from enum import Enum

logger = logging.getLogger("agentmesh.reliability.circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, tool_name: str, state: CircuitState):
        self.tool_name = tool_name
        self.state = state
        super().__init__(
            f"Circuit breaker OPEN for tool '{tool_name}' — "
            f"downstream appears saturated, failing fast"
        )


class CircuitBreaker:
    """Per-tool circuit breaker with closed/open/half-open states.

    Args:
        tool_name: The tool this breaker protects (for logging/keying).
        failure_threshold: Consecutive failures before tripping open.
        recovery_timeout: Seconds to wait before probing recovery (half-open).
        half_open_max_calls: Max probe calls allowed in half-open state.
    """

    def __init__(
        self,
        tool_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.tool_name = tool_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._half_open_successes = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current state, accounting for recovery timeout transition."""
        if self._state == CircuitState.OPEN and self._last_failure_time is not None:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                # Auto-transition to half-open on next access
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._half_open_successes = 0
                logger.info(
                    "CIRCUIT_HALF_OPEN tool=%s — probing recovery after %.1fs",
                    self.tool_name,
                    self.recovery_timeout,
                )
        return self._state

    def allow_call(self) -> bool:
        """Check if a call is allowed under the current circuit state."""
        current = self.state
        if current == CircuitState.CLOSED:
            return True
        if current == CircuitState.OPEN:
            return False
        # HALF_OPEN: allow limited probe calls
        if self._half_open_calls < self.half_open_max_calls:
            self._half_open_calls += 1
            return True
        return False

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.half_open_max_calls:
                # All probes succeeded — close the circuit
                self._state = CircuitState.CLOSED
                self._consecutive_failures = 0
                logger.info(
                    "CIRCUIT_CLOSED tool=%s — recovery confirmed, all %d probes succeeded",
                    self.tool_name,
                    self.half_open_max_calls,
                )
        else:
            # Reset failure count on any success in CLOSED state
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._last_failure_time = time.monotonic()
        if self._state == CircuitState.HALF_OPEN:
            # Probe failed — back to open
            self._state = CircuitState.OPEN
            self._consecutive_failures += 1
            logger.warning(
                "CIRCUIT_REOPENED tool=%s — half-open probe failed, back to OPEN",
                self.tool_name,
            )
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "CIRCUIT_OPEN tool=%s — %d consecutive failures (threshold=%d)",
                    self.tool_name,
                    self._consecutive_failures,
                    self.failure_threshold,
                )

    def reset(self) -> None:
        """Force-reset the circuit to closed (for testing)."""
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self._half_open_successes = 0

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(tool={self.tool_name}, state={self.state.value}, "
            f"failures={self._consecutive_failures})"
        )


# ── Global registry of circuit breakers, keyed by tool name ──

_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    tool_name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    half_open_max_calls: int = 3,
) -> CircuitBreaker:
    """Get or create a circuit breaker for a tool.

    The breaker is keyed by tool name — the same breaker protects all
    Workflows calling the same tool. This is the cross-Workflow bulkhead.
    """
    if tool_name not in _breakers:
        _breakers[tool_name] = CircuitBreaker(
            tool_name=tool_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls,
        )
    return _breakers[tool_name]


def reset_all_breakers() -> None:
    """Reset all circuit breakers (for testing)."""
    for breaker in _breakers.values():
        breaker.reset()
    _breakers.clear()


def get_all_breaker_states() -> dict[str, dict]:
    """Get the state of all circuit breakers (for monitoring/CLI)."""
    return {
        name: {
            "state": b.state.value,
            "consecutive_failures": b.consecutive_failures,
            "failure_threshold": b.failure_threshold,
        }
        for name, b in _breakers.items()
    }
