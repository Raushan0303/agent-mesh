"""LangGraph StateGraph for the sourcing agent — Week 4: full 5-node graph.

Node types (deliberately separated):
  Research  — agentic (multi-tool sequence through the Tool Registry)
  Score     — deterministic function (NO LLM, pure ranking logic)
  Decide    — agentic (picks best supplier from scored list)
  Approve   — human checkpoint (interrupt() pauses until human approves)
  Confirm   — agentic (finalizes, returns result)

The Score node is plain deterministic code, not an LLM call — this is
intentional. Not every node should be an LLM call. Scoring is a pure
function of price, rating, and lead time; adding an LLM here would
make it non-reproducible and harder to debug.

The Approve node uses LangGraph's interrupt() — a structural checkpoint,
not a prompt asking the model to be careful. The graph genuinely suspends
until a human calls /approve, which sends Command(resume=approval_data).
"""

import logging

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from app.agentmesh.tool_registry import registry
from app.agents.sourcing_agent.state import AgentState

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


def _build_decide_prompt(
    item: str,
    quantity: int,
    budget: float,
    scored_suppliers: list[dict],
    past_decisions: list[dict],
) -> list:
    """Build the LLM prompt for the Decide node.

    Returns a list of LLMMessage objects: a system message defining the
    agent's role, and a user message with the supplier data + past decisions.
    """
    from app.agentmesh.llm import LLMMessage

    system = (
        "You are a sourcing agent that selects the best supplier for a procurement request. "
        "You are given a list of scored suppliers (sorted by best combined score) and "
        "optionally past sourcing decisions from memory. "
        "Your job is to:\n"
        "1. Select the best supplier from the list\n"
        "2. Explain your reasoning, considering price, rating, lead time, and past experience\n"
        "3. If a past decision with positive feedback exists for the same item, factor that in\n\n"
        "Respond in this exact format:\n"
        "SELECTED: <supplier_name>\n"
        "RATIONALE: <your reasoning in 2-3 sentences mentioning price, rating, and lead time>"
    )

    # Build supplier table
    supplier_lines = []
    for i, s in enumerate(scored_suppliers):
        supplier_lines.append(
            f"  {i+1}. {s['name']} — price=${s['price']}/unit, "
            f"rating={s['rating']}/5, lead_time={s['lead_time_days']}d"
        )
    suppliers_text = "\n".join(supplier_lines)

    # Build past decisions context
    past_text = "No past sourcing decisions found."
    if past_decisions:
        past_lines = []
        for pd in past_decisions[:3]:
            meta = pd.get("metadata", {})
            if isinstance(meta, str):
                import json
                meta = json.loads(meta)
            past_lines.append(
                f"  - {pd.get('content', '')[:100]}..."
            )
        past_text = "\n".join(past_lines)

    user = (
        f"Procurement Request:\n"
        f"  Item: {item}\n"
        f"  Quantity: {quantity}\n"
        f"  Budget: ${budget}/unit\n\n"
        f"Scored Suppliers (best first):\n{suppliers_text}\n\n"
        f"Past Sourcing Decisions (from memory):\n{past_text}\n\n"
        f"Select the best supplier and explain your reasoning."
    )

    return [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=user),
    ]


def _parse_llm_decision(content: str, scored_suppliers: list[dict]) -> tuple[dict, str]:
    """Parse the LLM response into (selected_supplier, rationale).

    Expected format:
      SELECTED: <supplier_name>
      RATIONALE: <reasoning>

    Falls back to scored[0] if parsing fails.
    """
    best = scored_suppliers[0]
    rationale = content.strip()

    selected = None
    for line in content.split("\n"):
        if line.strip().upper().startswith("SELECTED:"):
            name = line.split(":", 1)[1].strip()
            # Match to a supplier (case-insensitive)
            for s in scored_suppliers:
                if s["name"].lower() == name.lower():
                    selected = s
                    break
            break

    if selected is None:
        # Fallback: use the top-scored supplier
        selected = best

    # Extract rationale
    for line in content.split("\n"):
        if line.strip().upper().startswith("RATIONALE:"):
            rationale = line.split(":", 1)[1].strip()
            break

    return selected, rationale


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

    # Build the LLM prompt
    messages = _build_decide_prompt(item, quantity, budget, scored, past_decisions)

    # Call the LLM through the adapter interface
    # Trace context is propagated automatically by Temporal's OTel interceptor
    # — the LLM client injects traceparent from the active OTel context
    try:
        from app.agentmesh.llm import get_llm_client
        client = get_llm_client()
        response = await client.complete(
            messages=messages,
            temperature=0.0,
            max_tokens=1000,
        )
        selected, rationale = _parse_llm_decision(response.content, scored)
        logger.info(
            "DECIDE_LLM_COMPLETED selected=%s model=%s tokens=%s cost_usd=%.6f",
            selected["name"],
            response.model,
            response.usage.get("total_tokens", "?"),
            response.cost_usd,
        )
        return {
            "selected_supplier": selected,
            "decision_reason": rationale,
            "past_decisions": past_decisions,
            "cost_incurred": response.cost_usd,
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


# ── Node 4: Approve (human checkpoint via interrupt()) ──


def approve_node(state: AgentState) -> dict:
    """Approve node: human checkpoint via LangGraph's interrupt().

    This is a STRUCTURAL checkpoint, not a prompt. The graph genuinely
    suspends here — interrupt() pauses execution and waits for a human
    to call Command(resume=approval_data).

    The resume value is a dict with:
      - approved: bool
      - comment: str (optional)

    If the checkpointer's Postgres connection dropped at this moment,
    the graph would fail to save the checkpoint. On Worker restart,
    Temporal would retry the Activity, which would re-invoke the graph
    from the last checkpoint (before Approve), and interrupt() would
    fire again — the human would need to approve again. No data loss,
    but the approval needs to be re-given.
    """
    selected = state.get("selected_supplier", {})
    brief = state.get("brief", {})

    # Present the decision to the human for approval
    approval_request = {
        "item": brief.item if hasattr(brief, "item") else brief.get("item", ""),
        "quantity": brief.quantity if hasattr(brief, "quantity") else brief.get("quantity", 0),
        "supplier": selected.get("name", ""),
        "price": selected.get("price", 0),
        "lead_time_days": selected.get("lead_time_days", 0),
        "rating": selected.get("rating", 0),
        "total_cost": selected.get("price", 0) * (
            brief.quantity if hasattr(brief, "quantity") else brief.get("quantity", 0)
        ),
    }

    logger.info(
        "APPROVE_INTERRUPT_FIRED supplier=%s item=%s total_cost=%.2f",
        approval_request["supplier"],
        approval_request["item"],
        approval_request["total_cost"],
    )

    # interrupt() pauses the graph here. The human must call
    # /approve with a resume value to continue.
    approval = interrupt(approval_request)

    # approval is the value passed to Command(resume=...)
    approved = approval.get("approved", False)
    comment = approval.get("comment", "")

    logger.info(
        "APPROVE_RESUMED approved=%s comment=%s",
        approved,
        comment,
    )

    return {
        "approval_status": "approved" if approved else "rejected",
        "approval_comment": comment,
        "rejection_count": state.get("rejection_count", 0) + (0 if approved else 1),
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


def build_sourcing_graph(checkpointer=None):
    """Build the full 5-node LangGraph StateGraph for the sourcing agent.

    Topology:
        START → Research → (retry loop) → Score → Decide → Approve → Confirm → END
                                              ↓ (rejected)
                                             END

    Args:
        checkpointer: A LangGraph checkpointer (e.g., PostgresSaver) for
            persisting graph state. Required for interrupt() to work —
            without a checkpointer, interrupt() raises an error.
    """
    graph = StateGraph(AgentState)

    # Add all 5 nodes
    graph.add_node("research", research_node)
    graph.add_node("score", score_node)
    graph.add_node("decide", decide_node)
    graph.add_node("approve", approve_node)
    graph.add_node("confirm", confirm_node)

    # Edges
    graph.add_edge(START, "research")

    # Research → retry loop or Score or END
    graph.add_conditional_edges(
        "research",
        _should_retry_or_end,
        {"research": "research", "score": "score", END: END},
    )

    # Score → Decide → Approve
    graph.add_edge("score", "decide")
    graph.add_edge("decide", "approve")

    # Approve → Confirm (if approved), Research (if rejected, retry), or END (max retries)
    graph.add_conditional_edges(
        "approve",
        _after_approve,
        {"confirm": "confirm", "research": "research", END: END},
    )

    # Confirm → END
    graph.add_edge("confirm", END)

    return graph.compile(checkpointer=checkpointer)
