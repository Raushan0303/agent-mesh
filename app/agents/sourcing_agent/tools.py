import logging
import uuid

from app.agentmesh.reliability import (
    CircuitBreakerOpenError,
    get_circuit_breaker,
)
from app.agentmesh.tool_registry import ToolSpec, registry
from app.agents.sourcing_agent.fault_injection import maybe_inject
from app.agents.sourcing_agent.mock_marketplace import MOCK_SUPPLIERS, query_mock_marketplace
from app.agents.sourcing_agent.tool_schemas import (
    CheckSellerRatingInput,
    CheckSellerRatingOutput,
    CreatePurchaseOrderInput,
    CreatePurchaseOrderOutput,
    GetPriceQuoteInput,
    GetPriceQuoteOutput,
    InitiatePaymentInput,
    InitiatePaymentOutput,
    QuerySuppliersInput,
    QuerySuppliersOutput,
)

logger = logging.getLogger("agentmesh.sourcing_agent.tools")


# ── Tool implementations (agent-specific) ──
# Week 5: query_suppliers and get_price_quote are wrapped with the
# platform's circuit breaker. The breaker is keyed by tool name —
# it protects across Workflows, not just within one.


async def query_suppliers_impl(item: str, budget: float, quantity: int) -> dict:
    """Query the mock marketplace for matching suppliers.

    Week 5: wrapped with circuit breaker + fault injection.
    The breaker trips open after 5 consecutive failures across Workflows,
    protecting other Workflows from a saturated downstream.
    """
    breaker = get_circuit_breaker("query_suppliers")

    if not breaker.allow_call():
        # Circuit is open — fail fast, don't hit the downstream
        raise CircuitBreakerOpenError("query_suppliers", breaker.state)

    # Fault injection: maybe simulate a flaky downstream
    maybe_inject("query_suppliers")

    try:
        results = query_mock_marketplace(item, budget, quantity)
        breaker.record_success()
        return {"suppliers": results}
    except Exception as e:
        breaker.record_failure()
        raise e


async def get_price_quote_impl(supplier_name: str, item: str, quantity: int) -> dict:
    """Get a price quote from a specific supplier.

    Week 5: wrapped with circuit breaker + fault injection.
    """
    breaker = get_circuit_breaker("get_price_quote")

    if not breaker.allow_call():
        raise CircuitBreakerOpenError("get_price_quote", breaker.state)

    # Fault injection
    maybe_inject("get_price_quote")

    try:
        for s in MOCK_SUPPLIERS:
            if s["name"] == supplier_name and s["item"].lower() == item.lower():
                result = {
                    "supplier_name": s["name"],
                    "item": s["item"],
                    "unit_price": s["price"],
                    "total_price": s["price"] * quantity,
                    "lead_time_days": s["lead_time_days"],
                    "in_stock": True,
                }
                breaker.record_success()
                return result
        result = {
            "supplier_name": supplier_name,
            "item": item,
            "unit_price": 0.0,
            "total_price": 0.0,
            "lead_time_days": 0,
            "in_stock": False,
        }
        breaker.record_success()
        return result
    except Exception as e:
        breaker.record_failure()
        raise e


async def check_seller_rating_impl(supplier_name: str) -> dict:
    """Check the rating of a supplier."""
    for s in MOCK_SUPPLIERS:
        if s["name"] == supplier_name:
            return {
                "supplier_name": s["name"],
                "rating": s["rating"],
                "total_orders": 150,
                "on_time_rate": 0.92,
            }
    return {
        "supplier_name": supplier_name,
        "rating": 0.0,
        "total_orders": 0,
        "on_time_rate": 0.0,
    }


async def create_purchase_order_impl(
    supplier_name: str, item: str, quantity: int, unit_price: float
) -> dict:
    """Create a mock purchase order with idempotency key.

    The idempotency key is derived from the Temporal workflow ID + step name.
    If the Activity is retried (e.g., Worker crashed after PO creation but
    before reporting back to Temporal), this function finds the existing PO
    and returns it instead of creating a duplicate.
    """
    from temporalio import activity

    from app.agents.sourcing_agent.db import get_db_pool

    workflow_id = activity.info().workflow_id
    idempotency_key = f"{workflow_id}:create_po"

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Check if we already created this PO (from a previous Activity retry)
        existing = await conn.fetchrow(
            "SELECT po_id, status FROM sourcing_agent_purchase_orders WHERE idempotency_key = $1",
            idempotency_key,
        )
        if existing:
            logger.info(
                "IDEMPOTENT_HIT tool=create_purchase_order key=%s po_id=%s — returning existing",
                idempotency_key,
                existing["po_id"],
            )
            return {
                "po_id": existing["po_id"],
                "supplier_name": supplier_name,
                "status": existing["status"],
            }

        # Create the PO
        po_id = f"PO-{uuid.uuid4().hex[:8]}"
        await conn.execute(
            "INSERT INTO sourcing_agent_purchase_orders "
            "(idempotency_key, po_id, supplier_name, item, quantity, unit_price, status) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7)",
            idempotency_key,
            po_id,
            supplier_name,
            item,
            quantity,
            unit_price,
            "created",
        )
        logger.info(
            "IDEMPOTENT_CREATE tool=create_purchase_order key=%s po_id=%s",
            idempotency_key,
            po_id,
        )
        return {"po_id": po_id, "supplier_name": supplier_name, "status": "created"}


async def initiate_payment_impl(po_id: str, amount: float) -> dict:
    """Initiate a mock payment with idempotency key.

    Same pattern as create_purchase_order — if the Activity is retried,
    the existing payment record is returned instead of creating a duplicate.
    """
    from temporalio import activity

    from app.agents.sourcing_agent.db import get_db_pool

    workflow_id = activity.info().workflow_id
    idempotency_key = f"{workflow_id}:initiate_payment"

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Check if we already initiated this payment
        existing = await conn.fetchrow(
            "SELECT payment_id, status FROM sourcing_agent_payment_intents WHERE idempotency_key = $1",
            idempotency_key,
        )
        if existing:
            logger.info(
                "IDEMPOTENT_HIT tool=initiate_payment key=%s payment_id=%s — returning existing",
                idempotency_key,
                existing["payment_id"],
            )
            return {
                "payment_id": existing["payment_id"],
                "po_id": po_id,
                "status": existing["status"],
            }

        # Initiate the payment
        payment_id = f"PAY-{uuid.uuid4().hex[:8]}"
        await conn.execute(
            "INSERT INTO sourcing_agent_payment_intents "
            "(idempotency_key, payment_id, po_id, amount, status) "
            "VALUES ($1, $2, $3, $4, $5)",
            idempotency_key,
            payment_id,
            po_id,
            amount,
            "initiated",
        )
        logger.info(
            "IDEMPOTENT_CREATE tool=initiate_payment key=%s payment_id=%s",
            idempotency_key,
            payment_id,
        )
        return {"payment_id": payment_id, "po_id": po_id, "status": "initiated"}


# ── Register tools into the platform registry (agent-specific registration) ──


def register_sourcing_tools() -> None:
    """Register all sourcing agent tools into the platform Tool Registry.

    Called at import time from agents/sourcing_agent/__init__.py
    """
    registry.register(
        ToolSpec(
            name="query_suppliers",
            input_model=QuerySuppliersInput,
            output_model=QuerySuppliersOutput,
            timeout_seconds=10.0,
            idempotency_required=False,
        ),
        query_suppliers_impl,
    )
    registry.register(
        ToolSpec(
            name="get_price_quote",
            input_model=GetPriceQuoteInput,
            output_model=GetPriceQuoteOutput,
            timeout_seconds=10.0,
            idempotency_required=False,
        ),
        get_price_quote_impl,
    )
    registry.register(
        ToolSpec(
            name="check_seller_rating",
            input_model=CheckSellerRatingInput,
            output_model=CheckSellerRatingOutput,
            timeout_seconds=10.0,
            idempotency_required=False,
        ),
        check_seller_rating_impl,
    )
    registry.register(
        ToolSpec(
            name="create_purchase_order",
            input_model=CreatePurchaseOrderInput,
            output_model=CreatePurchaseOrderOutput,
            timeout_seconds=15.0,
            idempotency_required=True,
        ),
        create_purchase_order_impl,
    )
    registry.register(
        ToolSpec(
            name="initiate_payment",
            input_model=InitiatePaymentInput,
            output_model=InitiatePaymentOutput,
            timeout_seconds=15.0,
            idempotency_required=True,
        ),
        initiate_payment_impl,
    )
