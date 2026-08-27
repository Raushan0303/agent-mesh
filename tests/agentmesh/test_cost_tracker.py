"""Tests for the CostTracker.

Tests:
1. Under budget — cost accumulated, not over budget
2. Over budget — is_over_budget returns True
3. Cost from InferRoute — record_llm_call with cost_usd from InferRoute
4. Direct provider ($0 cost) — record_llm_call with cost_usd=0
5. Cost in result — summary() returns correct fields
6. Budget check — is_over_budget compares correctly
"""

import pytest

from app.agentmesh.llm.client import LLMResponse
from app.agentmesh.reliability.cost_tracker import CostTracker


def test_under_budget():
    """Cost accumulated but under budget — is_over_budget returns False."""
    tracker = CostTracker()
    response = LLMResponse(content="test", model="gpt-4o-mini", usage={"total_tokens": 100}, cost_usd=0.01)
    tracker.record_llm_call(response)

    assert tracker.total_usd == pytest.approx(0.01)
    assert tracker.is_over_budget(0.50) is False


def test_over_budget():
    """Cost exceeds budget — is_over_budget returns True."""
    tracker = CostTracker()
    response = LLMResponse(content="test", model="gpt-4o", usage={"total_tokens": 1000}, cost_usd=0.60)
    tracker.record_llm_call(response)

    assert tracker.total_usd == pytest.approx(0.60)
    assert tracker.is_over_budget(0.50) is True


def test_cost_from_inferroute():
    """Record cost from InferRoute — cost_usd comes from InferRoute's response."""
    tracker = CostTracker()
    response1 = LLMResponse(content="test", model="gpt-4o-mini", usage={"total_tokens": 500}, cost_usd=0.075)
    response2 = LLMResponse(content="test", model="gpt-4o-mini", usage={"total_tokens": 300}, cost_usd=0.045)

    tracker.record_llm_call(response1)
    tracker.record_llm_call(response2)

    assert tracker.total_usd == pytest.approx(0.12)
    assert tracker.total_tokens == 800


def test_direct_provider_zero_cost():
    """Direct provider call — cost_usd is 0, tracker accumulates nothing."""
    tracker = CostTracker()
    response = LLMResponse(content="test", model="gpt-4o", usage={"total_tokens": 1000}, cost_usd=0.0)
    tracker.record_llm_call(response)

    assert tracker.total_usd == 0.0
    assert tracker.total_tokens == 1000
    assert tracker.is_over_budget(0.50) is False


def test_summary():
    """Summary returns correct fields."""
    tracker = CostTracker()
    response = LLMResponse(content="test", model="gpt-4o-mini", usage={"total_tokens": 100}, cost_usd=0.01)
    tracker.record_llm_call(response)
    tracker.record_tool_call("query_suppliers", cost_usd=0.0)

    summary = tracker.summary()
    assert summary["cost_incurred"] == pytest.approx(0.01)
    assert summary["tokens_used"] == 100
    assert summary["call_count"] == 2
    assert len(summary["calls"]) == 2
    assert summary["calls"][0]["type"] == "llm"
    assert summary["calls"][1]["type"] == "tool"


def test_budget_check_boundary():
    """Budget check at exact boundary — equal is NOT over budget."""
    tracker = CostTracker()
    response = LLMResponse(content="test", model="gpt-4o", usage={"total_tokens": 100}, cost_usd=0.50)
    tracker.record_llm_call(response)

    # Exactly at budget — not over
    assert tracker.is_over_budget(0.50) is False
    # One cent over — over
    response2 = LLMResponse(content="test", model="gpt-4o", usage={"total_tokens": 10}, cost_usd=0.01)
    tracker.record_llm_call(response2)
    assert tracker.is_over_budget(0.50) is True
