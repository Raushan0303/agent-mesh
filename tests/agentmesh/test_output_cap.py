"""Tests for the output size cap.

Tests:
1. Under cap — no truncation
2. Over cap (string) — string field truncated, _truncated: true set
3. Over cap (list) — list field truncated, _truncated: true set
4. Over cap (nested) — largest fields truncated first, smaller preserved
5. Custom cap — truncation happens at custom threshold
6. Empty result — no truncation, no error
"""

import json

import pytest

from app.agentmesh.tool_registry.output_cap import enforce_output_cap


def test_under_cap():
    """Result under the cap — no truncation."""
    result = {"name": "ACME Corp", "price": 10.50}
    capped = enforce_output_cap(result, max_bytes=65536)
    assert capped == result
    assert capped.get("_truncated") is None


def test_over_cap_string():
    """String field over the cap — truncated, _truncated: true set."""
    big_string = "A" * 100_000  # 100KB string
    result = {"name": "ACME Corp", "data": big_string}
    capped = enforce_output_cap(result, max_bytes=1024)

    assert capped["_truncated"] is True
    assert capped["_original_size_bytes"] > 1024
    assert capped["_max_bytes"] == 1024
    assert len(capped["data"]) < len(big_string)
    assert "[truncated]" in capped["data"]


def test_over_cap_list():
    """List field over the cap — truncated, _truncated: true set."""
    big_list = [{"id": i, "name": f"item_{i}"} for i in range(10_000)]
    result = {"items": big_list}
    capped = enforce_output_cap(result, max_bytes=1024)

    assert capped["_truncated"] is True
    assert len(capped["items"]) < len(big_list)
    assert "more items truncated" in capped["items"][-1]


def test_over_cap_nested():
    """Nested dict over the cap — largest fields truncated first."""
    result = {
        "small_field": "small value",
        "big_field": "B" * 50_000,
        "nested": {
            "inner_big": "C" * 50_000,
            "inner_small": "tiny",
        },
    }
    capped = enforce_output_cap(result, max_bytes=2048)

    assert capped["_truncated"] is True
    # Small field should be preserved
    assert capped["small_field"] == "small value"


def test_custom_cap():
    """Custom cap — truncation happens at the custom threshold."""
    result = {"data": "X" * 5000}  # 5KB
    capped = enforce_output_cap(result, max_bytes=1024)

    assert capped["_truncated"] is True
    assert capped["_max_bytes"] == 1024

    # Same result under a larger cap — no truncation
    capped_large = enforce_output_cap(result, max_bytes=65536)
    assert capped_large.get("_truncated") is None


def test_empty_result():
    """Empty result — no truncation, no error."""
    result = {}
    capped = enforce_output_cap(result, max_bytes=65536)
    assert capped == {}
    assert capped.get("_truncated") is None


def test_exact_boundary():
    """Result exactly at the cap — no truncation."""
    # Build a result that's exactly at the boundary
    payload = '{"x": "' + "A" * 100 + '"}'
    result = {"x": "A" * 100}
    capped = enforce_output_cap(result, max_bytes=len(payload.encode("utf-8")))
    assert capped.get("_truncated") is None
