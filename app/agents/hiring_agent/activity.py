"""Temporal Activities for the hiring agent — same two-Activity split as
sourcing_agent/activity.py:

  run_graph_until_interrupt  -> runs Screen Resume -> Score -> (Reject|Schedule)
                                 -> Interview (follow-up loop) -> Human Review
                                 returns when interrupt() fires (or graph ends)
  resume_graph                -> resumes with Command(resume=review_data)
                                 runs Human Review (resumed) -> Offer Decision -> END

The graph state is persisted in Postgres via the LangGraph checkpointer,
keyed by thread_id = workflow_id — same durability story as sourcing.
"""

import logging

from temporalio import activity

from app.agents.hiring_agent.state import HiringBriefInput

logger = logging.getLogger("agentmesh.hiring_agent.activity")


@activity.defn
async def run_graph_until_interrupt(brief: HiringBriefInput) -> dict:
    """Activity 1: run the graph until interrupt() fires or the graph ends.

    Returns a dict with:
      - paused: bool — True if paused at Human Review, False if it ended
      - review_request: dict — data presented to the recruiter
      - status: str — "paused" | "rejected_by_score"
      - screening_score / interview_score / skills_matched
    """
    from app.agents.hiring_agent.checkpointer import get_checkpointer
    from app.agents.hiring_agent.graph import build_hiring_graph

    checkpointer = await get_checkpointer()
    graph = build_hiring_graph(checkpointer=checkpointer)

    thread_id = activity.info().workflow_id
    config = {"configurable": {"thread_id": thread_id}}

    final_state = await graph.ainvoke(
        {
            "brief": brief,
            "screening_feedback": "",
            "followup_count": 0,
            "interview_score": 50.0,
            "interview_transcript": [],
            "rejection_count": 0,
        },
        config,
    )

    final_status = final_state.get("final_status", "")
    approval_status = final_state.get("approval_status", "")

    # Graph ended without pausing — either rejected by score, or (edge case,
    # same simplification sourcing makes) ran out of retries at Human Review.
    if final_status == "rejected_by_score":
        return {
            "paused": False,
            "review_request": None,
            "status": "rejected_by_score",
            "screening_score": final_state.get("screening_score", 0.0),
            "interview_score": None,
            "skills_matched": final_state.get("skills_matched", []),
        }

    if final_status:
        # offer_decision already ran inside this ainvoke (shouldn't normally
        # happen on the first pass — offer_decision only runs after resume —
        # but handled for completeness/symmetry with sourcing's activity).
        return {
            "paused": False,
            "review_request": None,
            "status": final_status,
            "screening_score": final_state.get("screening_score", 0.0),
            "interview_score": final_state.get("interview_score", 0.0),
            "skills_matched": final_state.get("skills_matched", []),
            "offer_amount": final_state.get("offer_amount"),
        }

    # Paused at Human Review — the approval_status is empty (not yet set)
    review_request = {
        "candidate_name": brief.candidate_name,
        "role": brief.role,
        "screening_score": final_state.get("screening_score", 0.0),
        "interview_score": final_state.get("interview_score", 0.0),
        "skills_matched": final_state.get("skills_matched", []),
        "interview_slot": final_state.get("interview_slot", {}),
    }

    logger.info(
        "GRAPH_PAUSED_AT_HUMAN_REVIEW thread_id=%s candidate=%s",
        thread_id,
        brief.candidate_name,
    )

    return {
        "paused": True,
        "review_request": review_request,
        "status": "paused",
        "screening_score": final_state.get("screening_score", 0.0),
        "interview_score": final_state.get("interview_score", 0.0),
        "skills_matched": final_state.get("skills_matched", []),
    }


@activity.defn
async def resume_graph(review_data: dict) -> dict:
    """Activity 2: resume the graph after the recruiter's decision.

    Resumes with Command(resume=review_data), unblocking interrupt() in
    Human Review. Runs Offer Decision -> END if approved; if rejected with
    retries left, the graph loops back to Screen Resume and may pause at
    Human Review again (same simplification sourcing makes — see
    graph.py's module docstring).
    """
    from langgraph.types import Command

    from app.agents.hiring_agent.checkpointer import get_checkpointer
    from app.agents.hiring_agent.graph import build_hiring_graph

    checkpointer = await get_checkpointer()
    graph = build_hiring_graph(checkpointer=checkpointer)

    thread_id = activity.info().workflow_id
    config = {"configurable": {"thread_id": thread_id}}

    logger.info(
        "GRAPH_RESUMING thread_id=%s approved=%s",
        thread_id,
        review_data.get("approved", False),
    )

    final_state = await graph.ainvoke(None, config, command=Command(resume=review_data))

    final_status = final_state.get("final_status", "awaiting_offer")
    approval_status = final_state.get("approval_status", "approved")

    logger.info(
        "GRAPH_COMPLETED thread_id=%s final_status=%s approval=%s",
        thread_id,
        final_status,
        approval_status,
    )

    return {
        "status": final_status,
        "approval_status": approval_status,
        "screening_score": final_state.get("screening_score", 0.0),
        "interview_score": final_state.get("interview_score", 0.0),
        "offer_amount": final_state.get("offer_amount"),
        "decision_reason": final_state.get("decision_reason", ""),
    }


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
