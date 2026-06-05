import pytest

from app.agents.sourcing_agent.graph import build_sourcing_graph, MAX_RESEARCH_ATTEMPTS
from app.agents.sourcing_agent.state import SourcingBriefInput


@pytest.mark.asyncio
async def test_research_finds_suppliers():
    """Normal case: suppliers found, status = completed."""
    graph = build_sourcing_graph()
    result = await graph.ainvoke(
        {
            "brief": SourcingBriefInput(item="USB-C cable", quantity=500, budget=3.00),
            "suppliers": [],
            "attempts": 0,
            "status": "running",
        }
    )
    assert result["status"] == "completed"
    assert len(result["suppliers"]) > 0
    assert result["attempts"] == 1


@pytest.mark.asyncio
async def test_research_no_matches_halts():
    """Edge case: impossible budget, halts after max attempts."""
    graph = build_sourcing_graph()
    result = await graph.ainvoke(
        {
            "brief": SourcingBriefInput(item="quantum computer", quantity=1, budget=0.01),
            "suppliers": [],
            "attempts": MAX_RESEARCH_ATTEMPTS - 1,
            "status": "running",
        }
    )
    assert result["status"] == "no_matches"
    assert result["attempts"] <= MAX_RESEARCH_ATTEMPTS
    assert result["suppliers"] == []


@pytest.mark.asyncio
async def test_research_hdmi_cable():
    """Another item: HDMI cable has suppliers in mock data."""
    graph = build_sourcing_graph()
    result = await graph.ainvoke(
        {
            "brief": SourcingBriefInput(item="HDMI cable", quantity=100, budget=5.00),
            "suppliers": [],
            "attempts": 0,
            "status": "running",
        }
    )
    assert result["status"] == "completed"
    assert len(result["suppliers"]) > 0
