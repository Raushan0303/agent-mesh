"""External outcome verification activities for the hiring agent — tri-state (task_2).

Same "never trust the client" principle as sourcing_agent/verification.py,
now with tri-state verification: VERIFIED / FAILED / UNKNOWN.

The critical distinction: FAILED means "safe to retry." UNKNOWN means
"NOT safe to retry — the offer may have been sent, we just can't confirm
it. Escalate to a human."
"""

import logging

from temporalio import activity

from app.agentmesh.verification import VerificationStatus, verification_result
from app.agents.hiring_agent.db import get_db_pool

logger = logging.getLogger("agentmesh.hiring_agent.verification")


@activity.defn
async def verify_offer_sent(
    offer_id: str,
    expected_candidate: str,
    expected_role: str,
    expected_amount: float,
) -> dict:
    """Verify that an offer was sent AND its fields are correct.

    Returns a tri-state result:
      VERIFIED  — offer exists and all fields match
      FAILED    — offer not found (safe to retry)
      UNKNOWN   — verification query failed (NOT safe to retry — escalate)
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT offer_id, candidate_name, role, offer_amount, status "
                "FROM hiring_agent_offers WHERE offer_id = $1",
                offer_id,
            )
    except Exception as e:
        logger.error(
            "VERIFICATION_UNKNOWN offer_id=%s — database query failed: %s",
            offer_id, e,
        )
        return verification_result(
            VerificationStatus.UNKNOWN,
            side_effect_id=offer_id,
            error=f"db_query_failed: {e}",
        )

    if row is None:
        logger.warning(
            "VERIFICATION_FAILED offer_id=%s — not found in database",
            offer_id,
        )
        return verification_result(
            VerificationStatus.FAILED,
            side_effect_id=offer_id,
            mismatches=["offer_not_found"],
        )

    mismatches = []
    if row["candidate_name"] != expected_candidate:
        mismatches.append(f"candidate_name: expected={expected_candidate} got={row['candidate_name']}")
    if row["role"] != expected_role:
        mismatches.append(f"role: expected={expected_role} got={row['role']}")
    if abs(row["offer_amount"] - expected_amount) > 0.01:
        mismatches.append(f"offer_amount: expected={expected_amount} got={row['offer_amount']}")
    if row["status"] != "sent":
        mismatches.append(f"status: expected=sent got={row['status']}")

    if mismatches:
        logger.warning(
            "VERIFICATION_FIELD_MISMATCH offer_id=%s mismatches=%s",
            offer_id, mismatches,
        )
        return verification_result(
            VerificationStatus.FAILED,
            side_effect_id=offer_id,
            mismatches=mismatches,
        )

    logger.info(
        "VERIFICATION_PASSED offer_id=%s candidate=%s role=%s amount=%.2f",
        offer_id, row["candidate_name"], row["role"], row["offer_amount"],
    )
    return verification_result(
        VerificationStatus.VERIFIED,
        side_effect_id=offer_id,
    )
