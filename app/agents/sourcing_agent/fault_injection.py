"""Fault-injection harness for the sourcing agent.

Delegates to the shared FaultInjector in app.agentmesh.reliability.
Kept as a thin wrapper so existing imports (set_failure_rate, maybe_inject, etc.)
continue to work without touching tool implementations or tests.
"""

from app.agentmesh.reliability.fault_injection import (
    FaultInjector,
    InjectedFailureError,
)

_injector = FaultInjector("sourcing_agent")

# Module-level API (backward-compatible)
set_failure_rate = _injector.set_failure_rate
get_failure_rate = _injector.get_failure_rate
clear_failure_rates = _injector.clear_failure_rates
maybe_inject = _injector.maybe_inject
is_fault_injection_enabled = _injector.is_enabled
