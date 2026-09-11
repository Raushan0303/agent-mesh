import logging
import uuid

from app.agentmesh.reliability import (
    CircuitBreakerOpenError,
    get_circuit_breaker,
)
from app.agentmesh.tool_registry import ToolSpec, registry
from app.agentmesh.tool_registry.spec import RiskTier
from app.agents.hiring_agent.fault_injection import maybe_inject
from app.agents.hiring_agent.mock_ats import schedule_slot
from app.agents.hiring_agent.tool_schemas import (
    ScheduleInterviewInput,
    ScheduleInterviewOutput,
    SendOfferInput,
    SendOfferOutput,
)

logger = logging.getLogger("agentmesh.hiring_agent.tools")


# ── Tool implementations (agent-specific) ──
# Same reliability pattern as sourcing_agent/tools.py: every tool call goes
# through the platform's circuit breaker (keyed by tool name, shared across
# Workflows) and can have fault injection enabled for chaos testing.


async def schedule_interview_impl(candidate_name: str, role: str) -> dict:
    """Book an interview slot through the mock calendar/ATS integration.

    Wrapped with circuit breaker + fault injection, same as sourcing's
    query_suppliers — this is a read/write call to an external system
    (the ATS) that can legitimately be flaky.
    """
    breaker = get_circuit_breaker("schedule_interview")

    if not breaker.allow_call():
        raise CircuitBreakerOpenError("schedule_interview", breaker.state)

    maybe_inject("schedule_interview")

    try:
        result = schedule_slot(candidate_name, role)
        breaker.record_success()
        return result
    except Exception as e:
        breaker.record_failure()
        raise e


async def send_offer_impl(candidate_name: str, role: str, offer_amount: float) -> dict:
    """Send an offer letter with an idempotency key.

    Same pattern as sourcing's create_purchase_order_impl — the idempotency
    key is derived from the Temporal workflow ID + step name. If the
    Activity is retried (Worker crashed after the offer email was sent but
    before Temporal recorded it), this finds the existing offer and returns
    it instead of sending a duplicate offer letter.
    """
    from temporalio import activity

    from app.agents.hiring_agent.db import get_db_pool

    breaker = get_circuit_breaker("send_offer")
    if not breaker.allow_call():
        raise CircuitBreakerOpenError("send_offer", breaker.state)
    maybe_inject("send_offer")

    workflow_id = activity.info().workflow_id
    idempotency_key = f"{workflow_id}:send_offer"

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT offer_id, status FROM hiring_agent_offers WHERE idempotency_key = $1",
            idempotency_key,
        )
        if existing:
            logger.info(
                "IDEMPOTENT_HIT tool=send_offer key=%s offer_id=%s — returning existing",
                idempotency_key,
                existing["offer_id"],
            )
            breaker.record_success()
            return {
                "offer_id": existing["offer_id"],
                "candidate_name": candidate_name,
                "status": existing["status"],
            }

        try:
            offer_id = f"OFFER-{uuid.uuid4().hex[:8]}"
            await conn.execute(
                "INSERT INTO hiring_agent_offers "
                "(idempotency_key, offer_id, candidate_name, role, offer_amount, status) "
                "VALUES ($1, $2, $3, $4, $5, $6)",
                idempotency_key,
                offer_id,
                candidate_name,
                role,
                offer_amount,
                "sent",
            )
            breaker.record_success()
            logger.info(
                "IDEMPOTENT_CREATE tool=send_offer key=%s offer_id=%s",
                idempotency_key,
                offer_id,
            )
            return {"offer_id": offer_id, "candidate_name": candidate_name, "status": "sent"}
        except Exception as e:
            breaker.record_failure()
            raise e


# ── Register tools into the platform registry (agent-specific registration) ──


def register_hiring_tools() -> None:
    """Register all hiring agent tools into the platform Tool Registry.

    Called at import time from agents/hiring_agent/__init__.py — same
    convention as register_sourcing_tools(). Both agents share one
    platform-wide MCP Tool Registry: schema-validated, sandboxed,
    circuit-broken tool calls, regardless of which agent owns the tool.
    """
    registry.register(
        ToolSpec(
            name="schedule_interview",
            input_model=ScheduleInterviewInput,
            output_model=ScheduleInterviewOutput,
            timeout_seconds=10.0,
            idempotency_required=False,
            risk_tier=RiskTier.SIDE_EFFECT,
        ),
        schedule_interview_impl,
    )
    registry.register(
        ToolSpec(
            name="send_offer",
            input_model=SendOfferInput,
            output_model=SendOfferOutput,
            timeout_seconds=15.0,
            idempotency_required=True,
            risk_tier=RiskTier.IRREVERSIBLE,
        ),
        send_offer_impl,
    )
