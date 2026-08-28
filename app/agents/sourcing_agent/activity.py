"""Temporal Activities for the sourcing agent.

Week 4: The graph is now a 5-node StateGraph with an interrupt() at the
Approve node. This requires two Activities:

  run_graph_until_interrupt  → runs Research → Score → Decide → Approve
                                returns when interrupt() fires (or graph completes)
  resume_graph               → resumes the graph with Command(resume=approval_data)
                                runs Approve (resumed) → Confirm → END

The graph state is persisted in Postgres via the LangGraph checkpointer,
keyed by thread_id = workflow_id. This means the graph's state survives
a Worker kill while paused at Approve — the checkpointer has the state,
and the resume Activity picks it up.
"""

import logging
from typing import Any

from temporalio import activity

from app.agents.sourcing_agent.state import (
    SourcingBriefInput,
    SourcingResult,
)

logger = logging.getLogger("agentmesh.sourcing_agent.activity")


@activity.defn
async def run_agent_graph(brief: SourcingBriefInput) -> dict:
    """Legacy Activity: run the full graph without interrupt support.

    Kept for backward compatibility with Week 1-3 tests.
    """
    from app.agents.sourcing_agent.graph import build_sourcing_graph
    from app.agents.sourcing_agent.checkpointer import get_checkpointer

    checkpointer = await get_checkpointer()
    graph = build_sourcing_graph(checkpointer=checkpointer)

    thread_id = activity.info().workflow_id
    config = {"configurable": {"thread_id": thread_id}}

    final_state = await graph.ainvoke(
        {
            "brief": brief,
            "trace_id": brief.trace_id or "",
            "suppliers": [],
            "attempts": 0,
            "status": "running",
        },
        config,
    )

    return {
        "suppliers": final_state.get("suppliers", []),
        "status": final_state.get("final_status", final_state.get("status", "failed")),
        "attempts": final_state.get("attempts", 0),
    }


@activity.defn
async def run_graph_until_interrupt(brief: SourcingBriefInput) -> dict:
    """Activity 1: run the graph until interrupt() fires or graph completes.

    Returns a dict with:
      - paused: bool — True if the graph paused at Approve, False if it completed
      - approval_request: dict — the data presented to the human for approval
      - suppliers: list — the suppliers found
      - status: str — "paused" | "completed" | "no_matches"
      - attempts: int
      - selected_supplier: dict — the supplier chosen by the Decide node
    """
    from app.agents.sourcing_agent.graph import build_sourcing_graph
    from app.agents.sourcing_agent.checkpointer import get_checkpointer

    checkpointer = await get_checkpointer()
    graph = build_sourcing_graph(checkpointer=checkpointer)

    thread_id = activity.info().workflow_id
    config = {"configurable": {"thread_id": thread_id}}

    # Invoke the graph. When interrupt() fires inside approve_node,
    # ainvoke() returns with the state up to that point.
    final_state = await graph.ainvoke(
        {
            "brief": brief,
            "suppliers": [],
            "attempts": 0,
            "status": "running",
        },
        config,
    )

    # Check if the graph paused at the Approve node
    # LangGraph stores interrupt info in the state
    status = final_state.get("status", "")
    final_status = final_state.get("final_status", "")
    approval_status = final_state.get("approval_status", "")

    # If the graph reached no_matches, it completed without pausing
    if status == "no_matches":
        return {
            "paused": False,
            "approval_request": None,
            "suppliers": [],
            "status": "no_matches",
            "attempts": final_state.get("attempts", 0),
            "selected_supplier": None,
            "cost_incurred": final_state.get("cost_incurred", 0.0),
        }

    # If the graph completed (e.g., auto-approved or rejected), return final
    if final_status:
        return {
            "paused": False,
            "approval_request": None,
            "suppliers": final_state.get("suppliers", []),
            "status": final_status,
            "attempts": final_state.get("attempts", 0),
            "selected_supplier": final_state.get("selected_supplier", {}),
            "cost_incurred": final_state.get("cost_incurred", 0.0),
        }

    # The graph paused at Approve — the approval_status is empty (not yet set)
    selected = final_state.get("selected_supplier", {})
    brief_data = final_state.get("brief", brief)

    approval_request = {
        "item": brief_data.item if hasattr(brief_data, "item") else brief_data.get("item", ""),
        "quantity": brief_data.quantity if hasattr(brief_data, "quantity") else brief_data.get("quantity", 0),
        "supplier": selected.get("name", ""),
        "price": selected.get("price", 0),
        "lead_time_days": selected.get("lead_time_days", 0),
        "rating": selected.get("rating", 0),
        "total_cost": selected.get("price", 0) * (
            brief_data.quantity if hasattr(brief_data, "quantity") else brief_data.get("quantity", 0)
        ),
    }

    logger.info(
        "GRAPH_PAUSED_AT_APPROVE thread_id=%s supplier=%s total_cost=%.2f",
        thread_id,
        approval_request["supplier"],
        approval_request["total_cost"],
    )

    return {
        "paused": True,
        "approval_request": approval_request,
        "suppliers": final_state.get("suppliers", []),
        "status": "paused",
        "attempts": final_state.get("attempts", 0),
        "selected_supplier": selected,
        "cost_incurred": final_state.get("cost_incurred", 0.0),
    }


@activity.defn
async def resume_graph(approval_data: dict) -> dict:
    """Activity 2: resume the graph after human approval.

    Called after the Workflow receives the "approve" Signal. Resumes the
    graph with Command(resume=approval_data), which unblocks the interrupt()
    in the Approve node. The graph then runs Confirm → END.

    Returns a dict with:
      - status: str — "completed" | "rejected"
      - selected_supplier: dict
      - approval_status: str — "approved" | "rejected"
    """
    from langgraph.types import Command

    from app.agents.sourcing_agent.graph import build_sourcing_graph
    from app.agents.sourcing_agent.checkpointer import get_checkpointer

    checkpointer = await get_checkpointer()
    graph = build_sourcing_graph(checkpointer=checkpointer)

    thread_id = activity.info().workflow_id
    config = {"configurable": {"thread_id": thread_id}}

    logger.info(
        "GRAPH_RESUMING thread_id=%s approved=%s",
        thread_id,
        approval_data.get("approved", False),
    )

    # Resume the graph with the human's approval decision
    final_state = await graph.ainvoke(
        None,  # None = continue from checkpoint
        config,
        command=Command(resume=approval_data),
    )

    final_status = final_state.get("final_status", "completed")
    approval_status = final_state.get("approval_status", "approved")
    selected = final_state.get("selected_supplier", {})

    logger.info(
        "GRAPH_COMPLETED thread_id=%s final_status=%s approval=%s",
        thread_id,
        final_status,
        approval_status,
    )

    return {
        "status": final_status,
        "selected_supplier": selected,
        "approval_status": approval_status,
        "cost_incurred": final_state.get("cost_incurred", 0.0),
    }


# ── Week 3 Activities (kept for backward compatibility) ──


@activity.defn
async def run_research_activity(brief: SourcingBriefInput) -> dict:
    """Week 3 Activity: run the research graph (query suppliers + quotes + ratings).

    Uses the aggressive retry policy — read-only tools are safe to retry.
    """
    from app.agents.sourcing_agent.graph import build_sourcing_graph

    graph = build_sourcing_graph()
    final_state = await graph.ainvoke(
        {
            "brief": brief,
            "suppliers": [],
            "attempts": 0,
            "status": "running",
        }
    )

    return {
        "suppliers": final_state.get("suppliers", []),
        "status": final_state.get("status", "failed"),
        "attempts": final_state.get("attempts", 0),
    }


@activity.defn
async def create_po_activity(
    supplier_name: str, item: str, quantity: int, unit_price: float
) -> dict:
    """Week 3 Activity: create a purchase order with idempotency."""
    from app.agentmesh.tool_registry import registry

    result = await registry.call(
        "create_purchase_order",
        {
            "supplier_name": supplier_name,
            "item": item,
            "quantity": quantity,
            "unit_price": unit_price,
        },
    )
    return {
        "po_id": result["po_id"],
        "supplier_name": result["supplier_name"],
        "status": result["status"],
    }


@activity.defn
async def initiate_payment_activity(
    po_id: str, amount: float
) -> dict:
    """Week 3 Activity: initiate payment with idempotency."""
    from app.agentmesh.tool_registry import registry

    result = await registry.call(
        "initiate_payment",
        {
            "po_id": po_id,
            "amount": amount,
        },
    )
    return {
        "payment_id": result["payment_id"],
        "po_id": result["po_id"],
        "status": result["status"],
    }


@activity.defn
async def score_suppliers_activity(suppliers: list[dict]) -> dict:
    """Week 3 versioned Activity: score and rank suppliers."""
    def score(s: dict) -> float:
        price = s.get("price", 999)
        rating = s.get("rating", 0)
        lead_time = s.get("lead_time_days", 999)
        rating_info = s.get("rating_info", {})
        on_time = rating_info.get("on_time_rate", 0.9)
        return price * 0.4 - rating * 0.3 - on_time * 0.2 + lead_time * 0.1

    scored = sorted(suppliers, key=score)
    return {"suppliers": scored}
