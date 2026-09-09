"""External outcome verification activities — tri-state (task_2).

These activities independently verify side effects by querying external state
— NOT the activity's return value. This is the "never trust the client"
principle applied to agent self-report.

Tri-state verification (task_2): verification can return three states:
  VERIFIED  — the side effect was confirmed (row exists, fields match)
  FAILED    — the side effect definitely didn't happen (row not found)
  UNKNOWN   — we can't tell (DB error, timeout, query failed)

The critical distinction: FAILED means "safe to retry." UNKNOWN means
"NOT safe to retry — escalate to a human with a recovery record."

- verify_po_exists: checks the PO database directly, including field-level
  correctness (supplier, item, quantity, unit_price — not just existence)
- verify_payment_initiated: checks the payment database directly, including
  amount correctness
"""

import logging

from temporalio import activity

from app.agentmesh.verification import VerificationStatus, verification_result
from app.agents.sourcing_agent.db import get_db_pool

logger = logging.getLogger("agentmesh.sourcing_agent.verification")


@activity.defn
async def verify_po_exists(
    po_id: str,
    expected_supplier: str | None = None,
    expected_item: str | None = None,
    expected_quantity: int | None = None,
    expected_unit_price: float | None = None,
) -> dict:
    """Verify that a purchase order exists AND its fields are correct.

    Returns a tri-state result:
      VERIFIED  — PO exists and all fields match
      FAILED    — PO not found in database (safe to retry)
      UNKNOWN   — verification query failed (NOT safe to retry — escalate)
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT po_id, status, supplier_name, item, quantity, unit_price "
                "FROM sourcing_agent_purchase_orders WHERE po_id = $1",
                po_id,
            )
    except Exception as e:
        logger.error(
            "VERIFICATION_UNKNOWN po_id=%s — database query failed: %s",
            po_id, e,
        )
        return verification_result(
            VerificationStatus.UNKNOWN,
            side_effect_id=po_id,
            error=f"db_query_failed: {e}",
        )

    if row is None:
        logger.warning(
            "VERIFICATION_FAILED po_id=%s — not found in database",
            po_id,
        )
        return verification_result(
            VerificationStatus.FAILED,
            side_effect_id=po_id,
            mismatches=["po_not_found"],
        )

    mismatches = []
    if expected_supplier is not None and row["supplier_name"] != expected_supplier:
        mismatches.append(f"supplier_name: expected={expected_supplier} got={row['supplier_name']}")
    if expected_item is not None and row["item"] != expected_item:
        mismatches.append(f"item: expected={expected_item} got={row['item']}")
    if expected_quantity is not None and row["quantity"] != expected_quantity:
        mismatches.append(f"quantity: expected={expected_quantity} got={row['quantity']}")
    if expected_unit_price is not None and abs(row["unit_price"] - expected_unit_price) > 0.01:
        mismatches.append(f"unit_price: expected={expected_unit_price} got={row['unit_price']}")

    if mismatches:
        logger.warning(
            "VERIFICATION_FAILED po_id=%s mismatches=%s",
            po_id, mismatches,
        )
        return verification_result(
            VerificationStatus.FAILED,
            side_effect_id=po_id,
            mismatches=mismatches,
            po_status=row["status"],
        )

    logger.info(
        "VERIFICATION_PASSED po_id=%s status=%s supplier=%s",
        po_id, row["status"], row["supplier_name"],
    )
    return verification_result(
        VerificationStatus.VERIFIED,
        side_effect_id=po_id,
        po_status=row["status"],
        supplier_name=row["supplier_name"],
    )


@activity.defn
async def verify_payment_initiated(
    payment_id: str,
    expected_amount: float | None = None,
) -> dict:
    """Verify that a payment was initiated AND its amount is correct.

    Returns a tri-state result:
      VERIFIED  — payment exists and amount matches
      FAILED    — payment not found (safe to retry)
      UNKNOWN   — verification query failed (NOT safe to retry — escalate)
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT payment_id, po_id, status, amount "
                "FROM sourcing_agent_payment_intents WHERE payment_id = $1",
                payment_id,
            )
    except Exception as e:
        logger.error(
            "VERIFICATION_UNKNOWN payment_id=%s — database query failed: %s",
            payment_id, e,
        )
        return verification_result(
            VerificationStatus.UNKNOWN,
            side_effect_id=payment_id,
            error=f"db_query_failed: {e}",
        )

    if row is None:
        logger.warning(
            "VERIFICATION_FAILED payment_id=%s — not found in database",
            payment_id,
        )
        return verification_result(
            VerificationStatus.FAILED,
            side_effect_id=payment_id,
            mismatches=["payment_not_found"],
        )

    mismatches = []
    if expected_amount is not None and abs(row["amount"] - expected_amount) > 0.01:
        mismatches.append(f"amount: expected={expected_amount} got={row['amount']}")

    if mismatches:
        logger.warning(
            "VERIFICATION_FIELD_MISMATCH payment_id=%s mismatches=%s",
            payment_id, mismatches,
        )
        return verification_result(
            VerificationStatus.FAILED,
            side_effect_id=payment_id,
            mismatches=mismatches,
            payment_status=row["status"],
        )

    logger.info(
        "VERIFICATION_PASSED payment_id=%s status=%s amount=%s",
        payment_id, row["status"], row["amount"],
    )
    return verification_result(
        VerificationStatus.VERIFIED,
        side_effect_id=payment_id,
        payment_status=row["status"],
        po_id=row["po_id"],
    )
