"""LangGraph StateGraph for the sourcing agent — stateless reasoning engine.

Node types (deliberately separated):
  Research         — agentic (multi-tool sequence through the Tool Registry)
  Score            — deterministic function (NO LLM, pure ranking logic)
  Decide           — agentic (picks best supplier from scored list)
  Process Approval — applies the human's decision (pure function)
  Confirm          — agentic (finalizes, returns result)

The Score node is plain deterministic code, not an LLM call — this is
intentional. Not every node should be an LLM call. Scoring is a pure
function of price, rating, and lead time; adding an LLM here would
make it non-reproducible and harder to debug.

STATELESS DESIGN (task_1 refactor):
  The graph no longer uses interrupt() or a checkpointer. Temporal owns
  100% of durable state. The pause point is a conditional edge:

    decide → _approval_gate → END      (no approval in state → pause)
                          → process_approval  (approval present → apply it)

  Phase 1 activity invokes with an initial state (no "approval" key).
  The graph runs to the gate and returns the FULL state dict — which
  Temporal records in its Event History. Phase 2 activity invokes with
  the same state + {"approval": {...}}; the START router sends it
  straight to process_approval. No thread_id, no ainvoke(None), no
  Command(resume=...), no second persistence engine.
"""

import logging

from langgraph.graph import StateGraph, START, END

from app.agentmesh.tool_registry import registry
from app.agents.sourcing_agent.state import AgentState
from app.agents.sourcing_agent.prompts import build_decide_prompt, parse_llm_decision

logger = logging.getLogger("agentmesh.sourcing_agent.graph")

MAX_RESEARCH_ATTEMPTS = 3


# ── Node 1: Research (agentic) ──


async def research_node(state: AgentState) -> dict:
    """Research node: multi-tool sequence through the registry.

    Sequence: query_suppliers → get_price_quote (per supplier) → check_seller_rating (per supplier)

    Stop condition:
    - If suppliers found → enrich with quotes + ratings, status = "completed"
    - If 3 attempts with no matches → status = "no_matches", return
    """
    brief = state["brief"]
    attempts = state.get("attempts", 0) + 1

    # Step 1: Query suppliers through the registry
    query_result = await registry.call(
        "query_suppliers",
        {
            "item": brief.item,
            "budget": brief.budget,
            "quantity": brief.quantity,
        },
    )
    suppliers = query_result.get("suppliers", [])

    if not suppliers:
        if attempts >= MAX_RESEARCH_ATTEMPTS:
            logger.info(
                "STOP_CONDITION_FIRED path=no_matches item=%s attempts=%d max_attempts=%d",
                brief.item,
                attempts,
                MAX_RESEARCH_ATTEMPTS,
            )
            return {"suppliers": [], "attempts": attempts, "status": "no_matches"}
        logger.info(
            "RESEARCH_RETRY item=%s attempts=%d/%d — no suppliers found, will retry",
            brief.item,
            attempts,
            MAX_RESEARCH_ATTEMPTS,
        )
        return {"suppliers": [], "attempts": attempts, "status": "running"}

    # Step 2: Enrich each supplier with a price quote + seller rating
    enriched = []
    for s in suppliers:
        quote = await registry.call(
            "get_price_quote",
            {
                "supplier_name": s["name"],
                "item": brief.item,
                "quantity": brief.quantity,
            },
        )
        rating = await registry.call(
            "check_seller_rating",
            {
                "supplier_name": s["name"],
            },
        )
        enriched.append(
            {
                "name": s["name"],
                "item": s["item"],
                "price": s["price"],
                "lead_time_days": s["lead_time_days"],
                "rating": s["rating"],
                "quote": quote,
                "rating_info": rating,
            }
        )

    logger.info(
        "STOP_CONDITION_FIRED path=completed item=%s attempts=%d suppliers_found=%d",
        brief.item,
        attempts,
        len(enriched),
    )
    return {"suppliers": enriched, "attempts": attempts, "status": "completed"}


# ── Node 2: Score (deterministic function — NO LLM) ──


def score_node(state: AgentState) -> dict:
    """Score node: deterministic ranking of suppliers.

    This is plain deterministic code — no LLM call, no randomness, no I/O.
    Given the same input suppliers, it always produces the same output order.
    This is tested by test_score_determinism.py.

    Scoring formula (lower score = better):
      score = price * 0.4 - rating * 0.3 - on_time_rate * 0.2 + lead_time * 0.1
    """
    suppliers = state.get("suppliers", [])

    def score(s: dict) -> float:
        price = s.get("price", 999)
        rating = s.get("rating", 0)
        lead_time = s.get("lead_time_days", 999)
        rating_info = s.get("rating_info", {})
        on_time = rating_info.get("on_time_rate", 0.9)
        # Lower score is better
        return price * 0.4 - rating * 0.3 - on_time * 0.2 + lead_time * 0.1

    scored = sorted(suppliers, key=score)
    logger.info(
        "SCORE_COMPLETED suppliers=%d best=%s",
        len(scored),
        scored[0]["name"] if scored else "none",
    )
    return {"scored_suppliers": scored}


# ── Node 3: Decide (agentic — LLM picks best supplier, queries memory) ──


async def decide_node(state: AgentState) -> dict:
    """Decide node: LLM picks the best supplier from the scored list.

    Week 6: queries the platform's memory store for past sourcing decisions
    on the same item. If a past decision exists with positive feedback,
    that context is passed to the LLM.

    Week 7: the LLM now produces the selection rationale, replacing the
    deterministic scored[0] pick. The LLM receives scored suppliers + past
    decisions and reasons about which one to select.

    The LLM call goes through the LLMClient adapter — today this calls
    OpenRouter/Groq/OpenAI directly. Tomorrow, swapping to InferRoute is
    a one-line config change (base_url), not a code change.
    """
    scored = state.get("scored_suppliers", [])
    if not scored:
        return {"selected_supplier": {}, "decision_reason": "no suppliers available"}

    brief = state.get("brief", {})
    item = brief.item if hasattr(brief, "item") else brief.get("item", "")
    quantity = brief.quantity if hasattr(brief, "quantity") else brief.get("quantity", 0)
    budget = brief.budget if hasattr(brief, "budget") else brief.get("budget", 0)

    # Query the memory store for past sourcing decisions
    past_decisions = []
    try:
        from app.agentmesh.memory.hybrid_retrieval import hybrid_search
        past_decisions = await hybrid_search(
            namespace="sourcing-agent",
            query=f"sourced {item}",
            top_k=3,
        )
    except Exception as e:
        logger.warning("MEMORY_QUERY_FAILED item=%s error=%s", item, e)

    # Build the LLM prompt (returns PromptVersion with hash + version)
    prompt = build_decide_prompt(item, quantity, budget, scored, past_decisions)

    # Call the LLM through the adapter interface
    # Trace context is propagated automatically by Temporal's OTel interceptor
    # — the LLM client injects traceparent from the active OTel context
    try:
        from app.agentmesh.llm import get_llm_client
        client = get_llm_client()
        response = await client.complete(
            messages=prompt.messages,
            temperature=0.0,
            max_tokens=1000,
        )
        selected, rationale = parse_llm_decision(response.content, scored)
        logger.info(
            "DECIDE_LLM_COMPLETED selected=%s model=%s tokens=%s cost_usd=%.6f prompt_hash=%s prompt_version=%s",
            selected["name"],
            response.model,
            response.usage.get("total_tokens", "?"),
            response.cost_usd,
            prompt.hash,
            prompt.version,
        )
        return {
            "selected_supplier": selected,
            "decision_reason": rationale,
            "past_decisions": past_decisions,
            "cost_incurred": response.cost_usd,
            "prompt_hash": prompt.hash,
            "prompt_version": prompt.version,
        }
    except Exception as e:
        logger.warning("DECIDE_LLM_FAILED error=%s — falling back to scored[0]", e)
        selected = scored[0]
        rationale = (
            f"Selected {selected['name']} — price=${selected['price']}, "
            f"rating={selected['rating']}, lead_time={selected['lead_time_days']}d"
        )

    logger.info(
        "DECIDE_COMPLETED selected=%s rationale_len=%d past_decisions=%d",
        selected["name"],
        len(rationale),
        len(past_decisions),
    )
    return {
        "selected_supplier": selected,
        "decision_reason": rationale,
        "past_decisions": past_decisions,
        "cost_incurred": 0.0,
    }


# ── Node 4: Process Approval (applies the human's decision) ──


def process_approval_node(state: AgentState) -> dict:
    """Process Approval node: apply the human's decision to state.

    STATELESS DESIGN: this node does NOT pause the graph. The pause is a
    conditional edge (_approval_gate): when the graph reaches it with no
    "approval" in state, the run ends and the full state returns to the
    Temporal Activity. Temporal's wait_condition owns the wait.

    When a phase-2 invocation comes in with state["approval"] set (the
    human's decision, injected by the workflow), the START router sends
    the graph straight here. This node applies the decision and CLEARS
    the approval key so a rejection-retry loop pauses again at the gate
    on the next pass.
    """
    approval = state.get("approval") or {}
    approved = approval.get("approved", False)
    comment = approval.get("comment", "")

    logger.info(
        "APPROVAL_APPLIED approved=%s comment=%s",
        approved,
        comment,
    )

    return {
        "approval_status": "approved" if approved else "rejected",
        "approval_comment": comment,
        "rejection_count": state.get("rejection_count", 0) + (0 if approved else 1),
        "approval": None,  # consume — next pass through the gate pauses again
    }


# ── Node 5: Confirm (agentic — finalizes) ──


def confirm_node(state: AgentState) -> dict:
    """Confirm node: finalize the result based on approval status.

    If approved → status = "completed", include selected supplier.
    If rejected → status = "rejected", no PO or payment.
    """
    approval = state.get("approval_status", "")
    selected = state.get("selected_supplier", {})

    if approval == "approved":
        logger.info(
            "CONFIRM_COMPLETED approved=true supplier=%s",
            selected.get("name", ""),
        )
        return {
            "final_status": "completed",
            "selected_supplier_name": selected.get("name", ""),
        }
    else:
        logger.info(
            "CONFIRM_REJECTED approved=false supplier=%s",
            selected.get("name", ""),
        )
        return {
            "final_status": "rejected",
            "selected_supplier_name": selected.get("name", ""),
        }


# ── Conditional edges ──


def _entry_router(state: AgentState) -> str:
    """START router: fresh run goes to Research; a resume invocation
    carrying the human's "approval" goes straight to Process Approval."""
    if state.get("approval") is not None:
        return "process_approval"
    return "research"


def _approval_gate(state: AgentState) -> str:
    """Approval gate after Decide: if no approval in state, end the run —
    the full state returns to the Temporal Activity (the pause). If the
    human's decision is present, apply it via Process Approval."""
    if state.get("approval") is None:
        return END
    return "process_approval"


def _should_retry_or_end(state: AgentState) -> str:
    """After Research: loop back if still running, else go to Score or end."""
    if state["status"] == "running":
        return "research"
    if state["status"] == "no_matches":
        return END
    return "score"


def _after_approve(state: AgentState) -> str:
    """After Approve: go to Confirm if approved, retry Research if rejected (with reason).

    On rejection, the agent loops back to Research with the human's feedback.
    This gives the agent a chance to find different suppliers or adjust its
    strategy based on the rejection reason. After MAX_REJECTIONS (default 2),
    the agent gives up and ends.
    """
    if state.get("approval_status") == "approved":
        return "confirm"

    # Rejection — check retry count
    rejection_count = state.get("rejection_count", 0) + 1
    if rejection_count >= 3:
        logger.info(
            "APPROVE_REJECTED_MAX_RETRIES rejections=%d — giving up",
            rejection_count,
        )
        return END

    logger.info(
        "APPROVE_REJECTED_RETRY rejection_count=%d comment=%s — going back to Research",
        rejection_count,
        state.get("approval_comment", ""),
    )
    return "research"


# ── Graph builder ──


def build_sourcing_graph():
    """Build the stateless LangGraph StateGraph for the sourcing agent.

    Topology:
        START → [_entry_router] → Research (fresh) | Process Approval (resume)
        Research → (retry loop) → Score → Decide
        Decide → [_approval_gate] → END (pause) | Process Approval
        Process Approval → Confirm (approved) | Research (rejected, retry) | END (maxed)
        Confirm → END

    No checkpointer, no interrupt() — the graph is a pure in-memory
    reasoning unit. Temporal owns the pause (wait_condition), the state
    (activity results in Event History), and the resume (a fresh
    invocation carrying the saved state + the human's approval).
    """
    graph = StateGraph(AgentState)

    # Add all 5 nodes
    graph.add_node("research", research_node)
    graph.add_node("score", score_node)
    graph.add_node("decide", decide_node)
    graph.add_node("process_approval", process_approval_node)
    graph.add_node("confirm", confirm_node)

    # START → Research (fresh) or Process Approval (resume with approval)
    graph.add_conditional_edges(
        START,
        _entry_router,
        {"research": "research", "process_approval": "process_approval"},
    )

    # Research → retry loop or Score or END
    graph.add_conditional_edges(
        "research",
        _should_retry_or_end,
        {"research": "research", "score": "score", END: END},
    )

    # Score → Decide → approval gate (pause at END, or apply approval)
    graph.add_edge("score", "decide")
    graph.add_conditional_edges(
        "decide",
        _approval_gate,
        {"process_approval": "process_approval", END: END},
    )

    # Process Approval → Confirm (approved), Research (rejected, retry), or END (max retries)
    graph.add_conditional_edges(
        "process_approval",
        _after_approve,
        {"confirm": "confirm", "research": "research", END: END},
    )

    # Confirm → END
    graph.add_edge("confirm", END)

    return graph.compile()
