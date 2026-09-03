"""Fault-injection harness for the hiring agent.

Delegates to the shared FaultInjector in app.agentmesh.reliability.
Scoped to this agent's own tool names (schedule_interview, send_offer)
so failure rates never collide across agents sharing the platform's tool registry.
"""

from app.agentmesh.reliability.fault_injection import (
    FaultInjector,
    InjectedFailureError,
)

_injector = FaultInjector("hiring_agent")

# Module-level API (backward-compatible)
set_failure_rate = _injector.set_failure_rate
get_failure_rate = _injector.get_failure_rate
clear_failure_rates = _injector.clear_failure_rates
maybe_inject = _injector.maybe_inject
is_fault_injection_enabled = _injector.is_enabled
