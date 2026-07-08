"""Agent-specific glue: wire the hiring agent's invoke path into the
platform's EvalHarness — same role as sourcing_agent/eval_glue.py.

The harness itself is agent-agnostic; this is the only hiring-specific
eval code. Runs Screen Resume -> Score -> (Reject | Schedule Interview),
stopping before the Interview/Human Review/Offer stages — same rationale
as sourcing's eval glue skipping Approve/Confirm: eval doesn't need a
human in the loop, it needs a fast, deterministic signal on whether the
agent's *gating* logic (screening + scoring) is correct.
"""

import asyncio
import logging

from app.agentmesh.evals.harness import harness
from app.agentmesh.evals.models import EvalScorecard
from app.agents.hiring_agent.eval_dataset import EVAL_DATASET

logger = logging.getLogger("agentmesh.hiring_agent.eval_glue")


async def invoke_hiring_agent(input: dict) -> dict:
    """Invoke the hiring agent for eval — runs Screen Resume + Score, and
    schedules the interview if the score clears the bar.

    Returns a dict with:
      - tool_calls: list of tool names called ("schedule_interview" if advanced)
      - status: "advanced_to_interview" | "rejected_by_score"
      - decision_rationale: the Screen Resume node's summary
    """
    from app.agents.hiring_agent.graph import schedule_interview_node, score_node, screen_resume_node
    from app.agents.hiring_agent.state import AgentState, HiringBriefInput

    brief = HiringBriefInput(**input)

    state: AgentState = {"brief": brief, "screening_feedback": ""}
    state.update(await screen_resume_node(state))
    state.update(score_node(state))

    tool_calls: list[str] = []
    if state["screening_score"] >= 55.0:
        state.update(await schedule_interview_node(state))
        tool_calls.append("schedule_interview")
        status = "advanced_to_interview"
    else:
        status = "rejected_by_score"

    return {
        "tool_calls": tool_calls,
        "status": status,
        "decision_rationale": state.get("screening_notes", ""),
        "screening_score": state.get("screening_score"),
        "skills_matched": state.get("skills_matched", []),
    }


async def run_eval_suite() -> EvalScorecard:
    """Run the full eval suite and return the scorecard.

    Same CI-gated entry point pattern as sourcing's run_eval_suite —
    fails the build if any score regresses.
    """
    logger.info("EVAL_SUITE_START scenarios=%d", len(EVAL_DATASET))

    results = await harness.run(EVAL_DATASET, invoke_hiring_agent)
    scorecard = harness.scorecard(results)

    passed, failures = harness.check_ci_gate(scorecard)

    print(f"\n{'='*60}")
    print("HIRING AGENT EVAL SCORECARD")
    print(f"{'='*60}")
    print(f"  Scenarios:          {scorecard.total_scenarios}")
    print(f"  Passed:             {scorecard.passed}/{scorecard.total_scenarios}")
    print(f"  Pass rate:          {scorecard.pass_rate:.2%}")
    print(f"  Avg tool accuracy:  {scorecard.avg_tool_accuracy:.2%}")
    print(f"  Completion rate:    {scorecard.completion_rate:.2%}")
    print(f"{'='*60}")

    if not passed:
        print(f"\n  CI GATE FAILED ❌")
        for f in failures:
            print(f"    - {f}")
    else:
        print(f"\n  CI GATE PASSED ✅")

    return scorecard


if __name__ == "__main__":
    asyncio.run(run_eval_suite())
