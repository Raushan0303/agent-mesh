"""
Circuit breaker state transition test.

Tests the closed → open → half-open → closed state machine, including:
- Trips open after N consecutive failures
- Fails fast when open (no downstream call)
- Transitions to half-open after recovery_timeout
- Closes after N successful probes in half-open
- Re-opens if a half-open probe fails

Also tests that the breaker is keyed by tool name — different tools
have independent breaker state.
"""

import asyncio
import time

import pytest

from app.agentmesh.reliability import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    get_circuit_breaker,
    reset_all_breakers,
)


def test_circuit_breaker_starts_closed():
    """A new circuit breaker starts in CLOSED state."""
    reset_all_breakers()
    breaker = get_circuit_breaker("test_tool_1", failure_threshold=3)
    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_call() is True


def test_circuit_breaker_trips_open_after_threshold():
    """After N consecutive failures, the breaker trips OPEN."""
    reset_all_breakers()
    breaker = get_circuit_breaker("test_tool_2", failure_threshold=3)

    # Record 2 failures — still closed
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED

    # 3rd failure — trips open
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_call() is False


def test_circuit_breaker_fails_fast_when_open():
    """When open, allow_call() returns False — no downstream call made."""
    reset_all_breakers()
    breaker = get_circuit_breaker("test_tool_3", failure_threshold=2)

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    # Should fail fast — no call allowed
    assert breaker.allow_call() is False


def test_circuit_breaker_success_resets_failure_count():
    """A success in CLOSED state resets the consecutive failure count."""
    reset_all_breakers()
    breaker = get_circuit_breaker("test_tool_4", failure_threshold=3)

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.consecutive_failures == 2

    breaker.record_success()
    assert breaker.consecutive_failures == 0
    assert breaker.state == CircuitState.CLOSED


def test_circuit_breaker_half_open_after_recovery_timeout():
    """After recovery_timeout, the breaker transitions to HALF_OPEN."""
    reset_all_breakers()
    breaker = get_circuit_breaker(
        "test_tool_5",
        failure_threshold=2,
        recovery_timeout=0.1,  # 100ms for fast testing
    )

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    # Wait for recovery timeout
    time.sleep(0.15)

    # Next access should transition to HALF_OPEN
    assert breaker.state == CircuitState.HALF_OPEN


def test_circuit_breaker_closes_after_successful_probes():
    """In HALF_OPEN, after N successful probes, the breaker closes."""
    reset_all_breakers()
    breaker = get_circuit_breaker(
        "test_tool_6",
        failure_threshold=2,
        recovery_timeout=0.1,
        half_open_max_calls=3,
    )

    # Trip open
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    # Wait for recovery
    time.sleep(0.15)
    assert breaker.state == CircuitState.HALF_OPEN

    # Make 3 successful probe calls
    assert breaker.allow_call() is True
    breaker.record_success()

    assert breaker.allow_call() is True
    breaker.record_success()

    assert breaker.allow_call() is True
    breaker.record_success()

    # All probes succeeded — circuit closes
    assert breaker.state == CircuitState.CLOSED
    assert breaker.consecutive_failures == 0


def test_circuit_breaker_reopens_on_half_open_failure():
    """If a half-open probe fails, the breaker re-opens."""
    reset_all_breakers()
    breaker = get_circuit_breaker(
        "test_tool_7",
        failure_threshold=2,
        recovery_timeout=0.1,
        half_open_max_calls=3,
    )

    # Trip open
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    # Wait for recovery
    time.sleep(0.15)
    assert breaker.state == CircuitState.HALF_OPEN

    # First probe succeeds
    assert breaker.allow_call() is True
    breaker.record_success()

    # Second probe fails — re-open
    assert breaker.allow_call() is True
    breaker.record_failure()

    assert breaker.state == CircuitState.OPEN


def test_circuit_breakers_keyed_by_tool_name_independent():
    """Different tools have independent circuit breaker state."""
    reset_all_breakers()
    breaker_a = get_circuit_breaker("tool_a", failure_threshold=2)
    breaker_b = get_circuit_breaker("tool_b", failure_threshold=2)

    # Trip breaker_a
    breaker_a.record_failure()
    breaker_a.record_failure()
    assert breaker_a.state == CircuitState.OPEN

    # breaker_b should still be closed
    assert breaker_b.state == CircuitState.CLOSED
    assert breaker_b.allow_call() is True


def test_circuit_breaker_integration_with_tool_impl():
    """Integration: calling query_suppliers_impl with a tripped breaker
    raises CircuitBreakerOpenError instead of hitting the downstream."""
    reset_all_breakers()
    from app.agents.sourcing_agent.tools import query_suppliers_impl
    from app.agents.sourcing_agent.fault_injection import clear_failure_rates

    clear_failure_rates()

    # Trip the breaker for query_suppliers
    breaker = get_circuit_breaker("query_suppliers", failure_threshold=2)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    # Calling the tool should raise CircuitBreakerOpenError
    with pytest.raises(CircuitBreakerOpenError):
        asyncio.run(query_suppliers_impl(item="USB-C cable", budget=3.00, quantity=500))

    # Reset for other tests
    reset_all_breakers()
