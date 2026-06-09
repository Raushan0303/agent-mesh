"""Test: 20-task eval harness scoring tool-selection accuracy.

Runs the sourcing agent's eval dataset through the platform's eval harness.
The harness is agent-agnostic; the dataset and invoke callback are agent-specific.
"""

import pytest

import app.agents.sourcing_agent  # noqa: F401 — triggers tool registration
from app.agentmesh.evals import EvalScenario, harness
from app.agents.sourcing_agent.eval_dataset import EVAL_DATASET
from app.agents.sourcing_agent.graph import build_sourcing_graph
from app.agents.sourcing_agent.state import SourcingBriefInput


async def invoke_sourcing_agent(input_data: dict) -> dict:
    """Invoke the sourcing agent graph and return output with tool_calls list.

    This is the agent-specific callback the harness calls.
    It runs the LangGraph graph and tracks which tools were called.
    """
    from app.agentmesh.tool_registry import registry

    # Track tool calls by wrapping the registry's call method
    original_call = registry.call
    tool_calls: list[str] = []

    async def tracking_call(name: str, args: dict) -> dict:
        tool_calls.append(name)
        return await original_call(name, args)

    registry.call = tracking_call  # type: ignore[assignment]
    try:
        brief = SourcingBriefInput(**input_data)
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
            "status": final_state.get("status", "failed"),
            "tool_calls": tool_calls,
            "suppliers": final_state.get("suppliers", []),
        }
    finally:
        registry.call = original_call  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_eval_dataset_tool_selection_accuracy():
    """Run the 20-scenario eval dataset and report tool-selection accuracy."""
    results = await harness.run(EVAL_DATASET, invoke_sourcing_agent)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    avg_tool_accuracy = sum(r.tool_accuracy for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"Eval Results: {passed}/{len(results)} passed, {failed} failed")
    print(f"Average tool-selection accuracy: {avg_tool_accuracy:.1%}")
    print(f"{'='*60}")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.scenario_id}: accuracy={r.tool_accuracy:.0%} tools={r.actual_tool_sequence}")
        if r.errors:
            for e in r.errors:
                print(f"         {e}")

    # Assert at least 80% pass rate (some edge cases may vary)
    assert passed >= 16, f"Expected at least 16/20 passes, got {passed}"
