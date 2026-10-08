"""Agent-specific glue: wire the sourcing agent's invoke path into the
platform's EvalHarness.

This is the ONLY agent-specific eval code — the harness itself is
agent-agnostic. The glue code:
  1. Takes an eval input (brief dict)
  2. Runs the sourcing agent's graph (Research → Score → Decide)
  3. Returns the result in the harness's expected format:
     {tool_calls, status, decision_rationale, retrieval_results}
"""

import asyncio
import logging

from app.agentmesh.evals.harness import harness
from app.agentmesh.evals.models import EvalScorecard
from app.agents.sourcing_agent.eval_dataset import EVAL_DATASET

logger = logging.getLogger("agentmesh.sourcing_agent.eval_glue")


async def invoke_sourcing_agent(input: dict) -> dict:
    """Invoke the sourcing agent for eval — runs the graph without the
    Approve/Confirm nodes (eval doesn't need human approval).

    Returns a dict with:
      - tool_calls: list of tool names called
      - status: "completed" | "no_matches"
      - decision_rationale: the Decide node's rationale (for LLM-as-judge)
      - retrieval_results: past decisions from memory (for RAG metrics)
    """
    from app.agents.sourcing_agent.graph import (
        research_node,
        score_node,
        decide_node,
    )
    from app.agents.sourcing_agent.state import AgentState, SourcingBriefInput

    # Build the brief
    brief = SourcingBriefInput(
        item=input["item"],
        quantity=input["quantity"],
        budget=input["budget"],
    )

    # Initialize state
    state: AgentState = {
        "brief": brief,
        "suppliers": [],
        "attempts": 0,
        "status": "running",
    }

    from app.agentmesh.tool_registry import registry

    # Record the tools ACTUALLY called (by the planner, through the
    # registry) — the eval scores these, not a list assumed from the outcome.
    with registry.trace_calls() as calls:
        for _ in range(4):  # the graph loops Research while status == "running"
            research_result = await research_node(state)
            state.update(research_result)
            if state["status"] != "running":
                break
        if state["status"] != "no_matches":
            state.update(score_node(state))
            state.update(await decide_node(state))

    tool_calls = []
    for c in calls:  # distinct tools, in order of first use
        if c["name"] not in tool_calls:
            tool_calls.append(c["name"])

    return {
        "tool_calls": tool_calls,
        "blocked_tool_calls": [c["name"] for c in calls if c.get("blocked")],
        "planner": state.get("planner", "unknown"),
        "status": state["status"],
        "decision_rationale": state.get("decision_reason", "") if state["status"] != "no_matches" else "",
        "retrieval_results": state.get("past_decisions", []),
        "selected_supplier": state.get("selected_supplier", {}),
    }


async def run_eval_suite() -> EvalScorecard:
    """Run the full eval suite and return the scorecard.

    This is the CI-gated entry point — called by the test suite and
    GitHub Actions. Fails the build if any score regresses.
    """
    logger.info("EVAL_SUITE_START scenarios=%d", len(EVAL_DATASET))

    results = await harness.run(EVAL_DATASET, invoke_sourcing_agent)
    scorecard = harness.scorecard(results)
    planners = sorted({r.actual_outcome.get("planner", "unknown") for r in results})
    blocked = sum(len(r.actual_outcome.get("blocked_tool_calls", [])) for r in results)

    # Check CI gate
    passed, failures = harness.check_ci_gate(scorecard)

    print(f"\n{'='*60}")
    print("EVAL SCORECARD")
    print(f"{'='*60}")
    print(f"  Scenarios:          {scorecard.total_scenarios}")
    print(f"  Passed:             {scorecard.passed}/{scorecard.total_scenarios}")
    print(f"  Pass rate:          {scorecard.pass_rate:.2%}")
    print(f"  Planner:            {', '.join(planners)}"
          + ("   (fixed = hard-coded tool order: tool accuracy is 100% by construction)"
             if any(p.startswith("fixed") for p in planners) else ""))
    print(f"  Avg tool accuracy:  {scorecard.avg_tool_accuracy:.2%}")
    print(f"  Blocked tool calls: {blocked} (side-effect tools chosen during research)")
    print(f"  Completion rate:    {scorecard.completion_rate:.2%}")
    if scorecard.avg_rag_recall is not None:
        print(f"  Avg RAG recall@5:   {scorecard.avg_rag_recall:.2%}")
    if scorecard.avg_judge_score is not None:
        print(f"  Avg judge score:    {scorecard.avg_judge_score:.2%}")
    print(f"{'='*60}")

    if not passed:
        print(f"\n  CI GATE FAILED ❌")
        for f in failures:
            print(f"    - {f}")
    else:
        print(f"\n  CI GATE PASSED ✅")

    scorecard.planner = ",".join(planners)
    return scorecard


if __name__ == "__main__":
    asyncio.run(run_eval_suite())
