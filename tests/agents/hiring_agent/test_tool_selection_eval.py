"""Test: 20-task eval harness scoring the hiring agent's screening gate.

Same structure as tests/agents/sourcing_agent/test_tool_selection_eval.py:
the harness is agent-agnostic; the dataset and invoke callback are
hiring-specific.
"""

import pytest

import app.agents.hiring_agent  # noqa: F401 — triggers tool registration
from app.agentmesh.evals import harness
from app.agents.hiring_agent.eval_dataset import EVAL_DATASET
from app.agents.hiring_agent.eval_glue import invoke_hiring_agent


@pytest.mark.asyncio
async def test_eval_dataset_tool_selection_accuracy():
    """Run the 20-scenario eval dataset and report tool-selection accuracy."""
    results = await harness.run(EVAL_DATASET, invoke_hiring_agent)

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

    assert passed >= 16, f"Expected at least 16/20 passes, got {passed}"


@pytest.mark.asyncio
async def test_eval_dataset_has_20_scenarios():
    assert len(EVAL_DATASET) == 20, f"Expected 20 scenarios, got {len(EVAL_DATASET)}"


@pytest.mark.asyncio
async def test_eval_dataset_scenario_ids_unique():
    ids = [s.scenario_id for s in EVAL_DATASET]
    assert len(ids) == len(set(ids)), "Duplicate scenario IDs found"
