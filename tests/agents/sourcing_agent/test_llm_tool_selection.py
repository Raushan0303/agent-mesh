"""Tool selection is now done by the model, so the eval can actually fail.

Before: research_node called query → quote → rating in a fixed order and
eval_glue reported a hard-coded tool list, so tool accuracy was 100% by
construction. Now, with a tool-calling client, the model picks the tools
and the eval scores the calls that really went through the registry.

These tests use scripted LLM clients (no network) to show that a good
planner scores 100%, and a bad one does NOT.
"""
import json

import pytest

import app.agents.sourcing_agent  # noqa: F401 — registers tools
from app.agentmesh.llm import LLMResponse, ToolCall
from app.agentmesh.llm import client as llm_module
from app.agentmesh.evals.harness import harness
from app.agentmesh.evals.models import EvalScenario
from app.agents.sourcing_agent.eval_glue import invoke_sourcing_agent

SCENARIO = EvalScenario(
    scenario_id="t-001",
    input={"item": "USB-C cable", "quantity": 100, "budget": 3.0},
    expected_tool_sequence=["query_suppliers", "get_price_quote", "check_seller_rating"],
    expected_outcome={"status": "completed"},
)


class ScriptedClient(llm_module.LLMClient):
    supports_tool_calling = True

    def __init__(self, policy):
        self.policy = policy
        self.n = 0

    async def complete(self, messages, model=None, temperature=0.0, max_tokens=1000):
        return LLMResponse(content="Selected the top-ranked supplier on price and rating.", model="scripted")

    async def complete_with_tools(self, messages, tools, model=None, temperature=0.0, max_tokens=1000):
        assert {t["function"]["name"] for t in tools} >= {"query_suppliers", "create_purchase_order"}
        self.n += 1
        calls = self.policy(messages)
        return LLMResponse(content="" if calls else "done", model="scripted",
                           tool_calls=[ToolCall(id=f"c{self.n}{i}", name=n, arguments=a) for i, (n, a) in enumerate(calls)])


def _last_suppliers(messages):
    for m in reversed(messages):
        if m["role"] == "tool":
            body = json.loads(m["content"])
            if "suppliers" in body:
                return [s["name"] for s in body["suppliers"]]
    return None


def good_policy(messages):
    names = _last_suppliers(messages)
    if names is None:
        return [("query_suppliers", {"item": "USB-C cable", "budget": 3.0, "quantity": 100})]
    if messages[-1]["role"] == "tool" and json.loads(messages[-1]["content"]).get("suppliers") is not None:
        return ([("get_price_quote", {"supplier_name": n, "item": "USB-C cable", "quantity": 100}) for n in names]
                + [("check_seller_rating", {"supplier_name": n}) for n in names])
    return []


def reckless_policy(messages):
    if _last_suppliers(messages) is None:
        return [("query_suppliers", {"item": "USB-C cable", "budget": 3.0, "quantity": 100})]
    if messages[-1]["role"] == "tool" and "suppliers" in json.loads(messages[-1]["content"]):
        return [("create_purchase_order", {"supplier_name": "SupplierAlpha", "item": "USB-C cable",
                                           "quantity": 100, "unit_price": 2.5})]
    return []


@pytest.fixture
def use_client():
    def install(policy):
        llm_module._client = ScriptedClient(policy)
    yield install
    llm_module.reset_llm_client()


@pytest.mark.asyncio
async def test_good_planner_scores_full_tool_accuracy(use_client):
    use_client(good_policy)
    [result] = await harness.run([SCENARIO], invoke_sourcing_agent)
    assert result.actual_outcome["planner"] == "llm"
    assert result.actual_tool_sequence == ["query_suppliers", "get_price_quote", "check_seller_rating"]
    assert result.tool_accuracy == 1.0 and result.passed


@pytest.mark.asyncio
async def test_reckless_planner_is_caught_and_never_executes_side_effects(use_client, monkeypatch):
    from app.agentmesh.tool_registry import registry

    async def boom(**kwargs):
        raise AssertionError("create_purchase_order must not execute during research")

    spec, _ = registry._tools["create_purchase_order"]
    monkeypatch.setitem(registry._tools, "create_purchase_order", (spec, boom))
    use_client(reckless_policy)
    [result] = await harness.run([SCENARIO], invoke_sourcing_agent)
    assert result.actual_outcome["blocked_tool_calls"] == ["create_purchase_order"]
    assert result.tool_accuracy < 1.0 and not result.passed


@pytest.mark.asyncio
async def test_planner_that_calls_no_tools_fails(use_client):
    use_client(lambda messages: [])
    [result] = await harness.run([SCENARIO], invoke_sourcing_agent)
    assert result.actual_tool_sequence == []
    assert result.tool_accuracy == 0.0 and not result.passed


@pytest.mark.asyncio
async def test_fixed_planner_is_labelled_as_such():
    llm_module._client = llm_module.DeterministicLLMClient()
    try:
        [result] = await harness.run([SCENARIO], invoke_sourcing_agent)
    finally:
        llm_module.reset_llm_client()
    assert result.actual_outcome["planner"] == "fixed"
