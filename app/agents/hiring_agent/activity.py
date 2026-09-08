"""Temporal Activities for the hiring agent — stateless graph design.

task_1 refactor: the LangGraph graph is now a pure in-memory reasoning
unit. There is ONE graph Activity instead of the old
run_graph_until_interrupt / resume_graph pair:

  run_hiring_graph(state: dict) -> dict

    Phase 1: workflow passes the initial state (brief set, no "approval").
      The graph runs Screen Resume → Score → Schedule → Interview →
      hits the review gate and returns the FULL state dict. Temporal
      records it in Event History — the checkpoint, owned by the
      orchestrator.

    Phase 2+: workflow injects the recruiter's decision into the state
      ({"approval": {...}}) and calls the same Activity again. The
      START router sends the graph straight to Process Review.

No checkpointer, no thread_id, no ainvoke(None), no Command(resume=...).
Temporal owns 100% of durable state; LangGraph owns zero.
"""

import logging

from temporalio import activity

from app.agents.hiring_agent.state import HiringBriefInput

logger = logging.getLogger("agentmesh.hiring_agent.activity")


def _normalize_state_out(state: dict) -> dict:
    """Make the graph state JSON-safe for the Temporal Activity boundary."""
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
        out["brief"] = HiringBriefInput(**brief)
    return out


def _build_review_request(state: dict) -> dict:
    """Build the recruiter-facing review request from the paused state."""
    brief = state.get("brief")
    candidate = brief.candidate_name if hasattr(brief, "candidate_name") else (brief or {}).get("candidate_name", "")
    role = brief.role if hasattr(brief, "role") else (brief or {}).get("role", "")

    return {
        "candidate_name": candidate,
        "role": role,
        "screening_score": state.get("screening_score", 0.0),
        "interview_score": state.get("interview_score", 0.0),
        "skills_matched": state.get("skills_matched", []),
        "interview_slot": state.get("interview_slot", {}),
    }


@activity.defn
async def run_hiring_graph(state: dict) -> dict:
    """Run the hiring graph statelessly — one Activity for both phases.

    Returns a dict with:
      - paused: bool — True if the graph stopped at the review gate
      - awaiting: str | None — "approval" when paused, else None
      - state: dict — the FULL graph state (JSON-safe) for the next phase
      - review_request: dict — recruiter-facing summary when paused
      - status / approval_status / screening_score / interview_score /
        skills_matched / offer_amount / decision_reason
    """
    from app.agents.hiring_agent.graph import build_hiring_graph

    graph = build_hiring_graph()  # no checkpointer — in-memory only

    final_state = await graph.ainvoke(_rehydrate_state_in(state))

    final_status = final_state.get("final_status", "")
    approval_status = final_state.get("approval_status", "")

    # Paused = run ended at the review gate: interview done, no terminal
    # status, no approval decision consumed yet.
    paused = not final_status and not approval_status

    if paused:
        derived_status = "paused"
    elif final_status:
        derived_status = final_status
    elif approval_status == "rejected":
        derived_status = "rejected"
    else:
        derived_status = "failed"

    thread_id = activity.info().workflow_id
    if paused:
        logger.info(
            "GRAPH_PAUSED_AT_REVIEW_GATE thread_id=%s candidate=%s",
            thread_id,
            _build_review_request(final_state).get("candidate_name", ""),
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
        "approval_status": approval_status,
        "rejection_count": final_state.get("rejection_count", 0),
        "screening_score": final_state.get("screening_score", 0.0),
        "interview_score": final_state.get("interview_score", 0.0),
        "skills_matched": final_state.get("skills_matched", []),
        "offer_amount": final_state.get("offer_amount"),
        "decision_reason": final_state.get("decision_reason", ""),
    }
    if paused:
        result["review_request"] = _build_review_request(final_state)
    return result


@activity.defn
async def send_offer_activity(candidate_name: str, role: str, offer_amount: float) -> dict:
    """Side-effecting Activity: send the offer letter, with idempotency.

    Kept outside the graph, same separation of concerns as sourcing's
    create_po_activity / initiate_payment_activity — agentic decisions
    live in the graph, side effects get their own Activity with a strict
    non-retryable retry template.
    """
    from app.agentmesh.tool_registry import registry

    result = await registry.call(
        "send_offer",
        {"candidate_name": candidate_name, "role": role, "offer_amount": offer_amount},
    )
    return {
        "offer_id": result["offer_id"],
        "candidate_name": result["candidate_name"],
        "status": result["status"],
    }
