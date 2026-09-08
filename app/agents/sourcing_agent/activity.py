"""Temporal Activities for the sourcing agent — stateless graph design.

task_1 refactor: the LangGraph graph is now a pure in-memory reasoning
unit. There is ONE graph Activity instead of the old
run_graph_until_interrupt / resume_graph pair:

  run_sourcing_graph(state: dict) -> dict

    Phase 1: workflow passes the initial state (brief set, no "approval").
      The graph runs Research → Score → Decide → hits the approval gate
      and returns the FULL state dict. Temporal records it in Event
      History — the checkpoint, owned by the orchestrator.

    Phase 2+: workflow injects the human's decision into the state
      ({"approval": {...}}) and calls the same Activity again. The
      START router sends the graph straight to Process Approval.

There is no checkpointer, no thread_id, no ainvoke(None), no
Command(resume=...). Temporal owns 100% of durable state; LangGraph
owns zero. If the worker dies while the workflow waits for approval,
the state is already in Temporal's Event History — a new worker
replays the workflow and re-issues the same state to this Activity.
"""

import logging

from temporalio import activity

from app.agents.sourcing_agent.state import (
    SourcingBriefInput,
    SourcingResult,
)

logger = logging.getLogger("agentmesh.sourcing_agent.activity")


def _normalize_state_out(state: dict) -> dict:
    """Make the graph state JSON-safe for the Temporal Activity boundary.

    The brief travels inside the graph as a SourcingBriefInput (nodes use
    attribute access). For the return trip across the Activity boundary
    it is converted to a plain dict — the next invocation rehydrates it.
    """
    out = dict(state)
    brief = out.get("brief")
    if hasattr(brief, "model_dump"):
        out["brief"] = brief.model_dump()
    return out


def _rehydrate_state_in(state: dict) -> dict:
    """Convert the serialized state back for the graph — brief dict → model."""
    out = dict(state)
    brief = out.get("brief")
    if isinstance(brief, dict):
        out["brief"] = SourcingBriefInput(**brief)
    return out


def _build_approval_request(state: dict) -> dict:
    """Build the human-facing approval request from the paused state."""
    selected = state.get("selected_supplier", {})
    brief = state.get("brief")
    item = brief.item if hasattr(brief, "item") else (brief or {}).get("item", "")
    quantity = brief.quantity if hasattr(brief, "quantity") else (brief or {}).get("quantity", 0)

    return {
        "item": item,
        "quantity": quantity,
        "supplier": selected.get("name", ""),
        "price": selected.get("price", 0),
        "lead_time_days": selected.get("lead_time_days", 0),
        "rating": selected.get("rating", 0),
        "total_cost": selected.get("price", 0) * quantity,
    }


@activity.defn
async def run_sourcing_graph(state: dict) -> dict:
    """Run the sourcing graph statelessly — one Activity for both phases.

    Returns a dict with:
      - paused: bool — True if the graph stopped at the approval gate
      - awaiting: str | None — "approval" when paused, else None
      - state: dict — the FULL graph state (JSON-safe) for the next phase
      - approval_request: dict — human-facing summary when paused
      - status / final_status / approval_status / attempts / suppliers /
        selected_supplier / cost_incurred / prompt_version / prompt_hash
    """
    from app.agents.sourcing_agent.graph import build_sourcing_graph

    graph = build_sourcing_graph()  # no checkpointer — in-memory only

    final_state = await graph.ainvoke(_rehydrate_state_in(state))

    status = final_state.get("status", "")
    final_status = final_state.get("final_status", "")
    approval_status = final_state.get("approval_status", "")

    # Paused = run ended at the approval gate: no terminal status and no
    # approval decision consumed yet. Everything else is a finished run.
    paused = not final_status and status != "no_matches" and not approval_status

    if paused:
        derived_status = "paused"
    elif final_status:
        derived_status = final_status
    elif approval_status == "rejected":
        derived_status = "rejected"
    elif status == "no_matches":
        derived_status = "no_matches"
    else:
        derived_status = status or "failed"

    thread_id = activity.info().workflow_id
    if paused:
        logger.info(
            "GRAPH_PAUSED_AT_APPROVAL_GATE thread_id=%s supplier=%s",
            thread_id,
            final_state.get("selected_supplier", {}).get("name", ""),
        )
    else:
        logger.info(
            "GRAPH_COMPLETED thread_id=%s status=%s approval=%s",
            thread_id,
            derived_status,
            approval_status,
        )

    result = {
        "paused": paused,
        "awaiting": "approval" if paused else None,
        "state": _normalize_state_out(final_state),
        "status": derived_status,
        "attempts": final_state.get("attempts", 0),
        "suppliers": final_state.get("suppliers", []),
        "selected_supplier": final_state.get("selected_supplier", {}),
        "approval_status": approval_status,
        "rejection_count": final_state.get("rejection_count", 0),
        "cost_incurred": final_state.get("cost_incurred", 0.0),
        "prompt_version": final_state.get("prompt_version", ""),
        "prompt_hash": final_state.get("prompt_hash", ""),
    }
    if paused:
        result["approval_request"] = _build_approval_request(final_state)
    return result


@activity.defn
async def run_agent_graph(brief: SourcingBriefInput) -> dict:
    """Legacy Activity: run the full graph from an initial state.

    Kept for backward compatibility with Week 1-3 tests. Stateless now —
    the run stops at the approval gate if it gets that far (no human
    approval can be injected through this entry point).
    """
    result = await run_sourcing_graph(
        {
            "brief": brief.model_dump(),
            "trace_id": getattr(brief, "trace_id", "") or "",
            "suppliers": [],
            "attempts": 0,
            "status": "running",
        }
    )
    return {
        "suppliers": result.get("suppliers", []),
        "status": result.get("status", "failed"),
        "attempts": result.get("attempts", 0),
    }


# ── Week 3 Activities (kept for backward compatibility) ──


@activity.defn
async def run_research_activity(brief: SourcingBriefInput) -> dict:
    """Week 3 Activity: run the research graph (query suppliers + quotes + ratings).

    Uses the aggressive retry policy — read-only tools are safe to retry.
    """
    result = await run_sourcing_graph(
        {
            "brief": brief.model_dump(),
            "suppliers": [],
            "attempts": 0,
            "status": "running",
        }
    )
    return {
        "suppliers": result.get("suppliers", []),
        "status": result.get("status", "failed"),
        "attempts": result.get("attempts", 0),
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
