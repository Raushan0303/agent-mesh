"""External outcome verification activities.

These activities independently verify side effects by querying external state
— NOT the activity's return value. This is the "never trust the client"
principle applied to agent self-report.

- verify_po_exists: checks the PO database directly
- verify_payment_initiated: checks the payment database directly

If the activity said it created a PO but the PO isn't in the database
(bug, DB rollback, network partition), verification fails — regardless
of what the activity reported.
"""

import logging

from temporalio import activity

from app.agents.sourcing_agent.db import get_db_pool

logger = logging.getLogger("agentmesh.sourcing_agent.verification")


@activity.defn
async def verify_po_exists(po_id: str) -> dict:
    """Verify that a purchase order actually exists in the PO database.

    Queries the PO database directly — does NOT trust create_po_activity's
    return value.

    Returns:
        {"verified": bool, "po_id": str, "po_status": str | None}
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT po_id, status, supplier_name, item, quantity, unit_price "
            "FROM sourcing_agent_purchase_orders WHERE po_id = $1",
            po_id,
        )

    if row is None:
        logger.warning(
            "VERIFICATION_FAILED po_id=%s — not found in database",
            po_id,
        )
        return {"verified": False, "po_id": po_id, "po_status": None}

    logger.info(
        "VERIFICATION_PASSED po_id=%s status=%s supplier=%s",
        po_id, row["status"], row["supplier_name"],
    )
    return {
        "verified": True,
        "po_id": po_id,
        "po_status": row["status"],
        "supplier_name": row["supplier_name"],
    }


@activity.defn
async def verify_payment_initiated(payment_id: str) -> dict:
    """Verify that a payment was actually initiated in the payment database.

    Queries the payment database directly — does NOT trust
    initiate_payment_activity's return value.

    Returns:
        {"verified": bool, "payment_id": str, "payment_status": str | None}
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT payment_id, po_id, status, amount "
            "FROM sourcing_agent_payment_intents WHERE payment_id = $1",
            payment_id,
        )

    if row is None:
        logger.warning(
            "VERIFICATION_FAILED payment_id=%s — not found in database",
            payment_id,
        )
        return {"verified": False, "payment_id": payment_id, "payment_status": None}

    logger.info(
        "VERIFICATION_PASSED payment_id=%s status=%s amount=%s",
        payment_id, row["status"], row["amount"],
    )
    return {
        "verified": True,
        "payment_id": payment_id,
        "payment_status": row["status"],
        "po_id": row["po_id"],
    }
