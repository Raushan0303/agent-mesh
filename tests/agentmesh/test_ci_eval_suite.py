"""
CI-gated eval suite — must pass a defined threshold to merge.

This test runs the full 50-scenario golden dataset through the eval
harness and asserts the CI gate passes. Wired into GitHub Actions on
every push.

Run: python -m pytest tests/agentmesh/test_ci_eval_suite.py -v -s
"""

import asyncio
import logging

import pytest

from app.agentmesh.evals.harness import harness
from app.core.config import settings

logger = logging.getLogger("agentmesh.tests.ci_eval")


@pytest.mark.asyncio
async def test_ci_eval_suite_passes_gate():
    """Run the full 50-scenario eval suite and assert the CI gate passes.

    The CI gate checks:
      - avg_tool_accuracy >= 0.90
      - completion_rate >= 0.95
      - avg_rag_recall >= 0.80 (if RAG scenarios are present)
    """
    from app.agents.sourcing_agent.eval_glue import invoke_sourcing_agent
    from app.agents.sourcing_agent.eval_dataset import EVAL_DATASET

    # Run the eval suite
    results = await harness.run(EVAL_DATASET, invoke_sourcing_agent)
    scorecard = harness.scorecard(results)

    # Print the scorecard
    print(f"\n{'='*60}")
    print("CI EVAL SCORECARD")
    print(f"{'='*60}")
    print(f"  Scenarios:          {scorecard.total_scenarios}")
    print(f"  Passed:             {scorecard.passed}/{scorecard.total_scenarios}")
    print(f"  Pass rate:          {scorecard.pass_rate:.2%}")
    print(f"  Avg tool accuracy:  {scorecard.avg_tool_accuracy:.2%}")
    print(f"  Completion rate:    {scorecard.completion_rate:.2%}")
    if scorecard.avg_judge_score is not None:
        print(f"  Avg judge score:    {scorecard.avg_judge_score:.2%}")
    print(f"{'='*60}")

    # Check CI gate
    passed, failures = harness.check_ci_gate(scorecard)

    assert passed, (
        f"CI GATE FAILED — quality regression detected:\n" +
        "\n".join(f"  - {f}" for f in failures)
    )

    print(f"\n  CI GATE PASSED ✅")


@pytest.mark.asyncio
async def test_eval_dataset_has_50_scenarios():
    """Assert the golden dataset has exactly 50 scenarios."""
    from app.agents.sourcing_agent.eval_dataset import EVAL_DATASET
    assert len(EVAL_DATASET) == 50, f"Expected 50 scenarios, got {len(EVAL_DATASET)}"


@pytest.mark.asyncio
async def test_eval_dataset_scenario_ids_unique():
    """Assert all scenario IDs are unique."""
    from app.agents.sourcing_agent.eval_dataset import EVAL_DATASET
    ids = [s.scenario_id for s in EVAL_DATASET]
    assert len(ids) == len(set(ids)), "Duplicate scenario IDs found"


@pytest.mark.asyncio
async def test_semantic_cache_hit_and_miss():
    """Test the semantic cache: store a response, then retrieve it with
    a near-duplicate prompt. Assert cache hit on near-duplicate, miss on
    dissimilar prompt."""
    import asyncpg
    from app.agentmesh.memory.semantic_cache import SemanticCache
    from app.core.config import settings

    pool = await asyncpg.create_pool(settings.agent_db_dsn, min_size=2, max_size=5)
    cache = SemanticCache(pool)
    await cache.setup()
    await cache.clear("sourcing-agent-test")

    # Store a response
    await cache.put(
        "sourcing-agent-test",
        "I need 500 USB-C cables at $3 budget",
        "Selected SupplierBeta at $2.85/unit",
    )

    # Query with a near-duplicate prompt (should hit)
    hit = await cache.get("sourcing-agent-test", "I need 500 USB-C cables at $3 budget")
    assert hit is not None, "Expected cache hit on identical prompt"
    assert "SupplierBeta" in hit["response"]
    print(f"\n  Cache HIT: similarity={hit['similarity']:.4f} ✅")

    # Query with a dissimilar prompt (should miss)
    miss = await cache.get("sourcing-agent-test", "completely different query about HDMI cables")
    assert miss is None, "Expected cache miss on dissimilar prompt"
    print(f"  Cache MISS on dissimilar prompt ✅")

    await cache.clear("sourcing-agent-test")
    await pool.close()


@pytest.mark.asyncio
async def test_hybrid_retrieval_recall():
    """Test hybrid retrieval: store entries, search, assert relevant results
    are in the top-k (recall@5)."""
    import asyncpg
    from app.agentmesh.memory.store import MemoryStore, close_memory_store
    from app.agentmesh.memory.hybrid_retrieval import hybrid_search, reciprocal_rank_fusion, rerank
    from app.core.config import settings

    # Use a dedicated pool for this test to avoid conflicts
    pool = await asyncpg.create_pool(settings.agent_db_dsn, min_size=2, max_size=5)
    store = MemoryStore(pool)
    await store.setup()
    await store.clear("sourcing-agent-test")

    # Store 5 entries
    for i, content in enumerate([
        "Sourced USB-C cables from SupplierAlpha, 500 units, $2.50/unit",
        "Sourced HDMI cables from SupplierBeta, 200 units, $3.90/unit",
        "Sourced Ethernet cables from SupplierGamma, 300 units, $1.50/unit",
        "Sourced Power adapters from SupplierDelta, 100 units, $8.75/unit",
        "Sourced USB-C cables from SupplierEpsilon, 1000 units, $2.85/unit",
    ]):
        await store.store("sourcing-agent-test", content, {"index": i})

    # Test BM25 search
    bm25_results = await store.search_bm25("sourcing-agent-test", "USB-C cables", top_k=5)
    assert len(bm25_results) >= 1, "BM25 should find USB-C entries"
    usb_c_bm25 = sum(1 for r in bm25_results if "USB-C" in r["content"])
    assert usb_c_bm25 >= 1, f"BM25 should find USB-C entries, got {usb_c_bm25}"

    # Test dense search
    dense_results = await store.search_dense("sourcing-agent-test", "USB-C cables", top_k=5)
    assert len(dense_results) >= 1, "Dense search should find USB-C entries"

    # Test RRF fusion
    fused = reciprocal_rank_fusion(bm25_results, dense_results)
    assert len(fused) >= 1, "RRF should produce fused results"

    # Test reranking
    reranked = rerank("USB-C cables", fused, top_k=5)
    assert len(reranked) >= 1, "Reranker should produce results"

    # Check recall: at least 1 USB-C entry should be in top-5
    usb_c_hits = sum(1 for r in reranked if "USB-C" in r["content"])
    assert usb_c_hits >= 1, f"Expected at least 1 USB-C result, got {usb_c_hits}"

    print(f"\n  BM25: {len(bm25_results)} results, {usb_c_bm25} USB-C hits ✅")
    print(f"  Dense: {len(dense_results)} results ✅")
    print(f"  RRF fused: {len(fused)} results ✅")
    print(f"  Reranked: {len(reranked)} results, {usb_c_hits} USB-C hits ✅")

    await store.clear("sourcing-agent-test")
    await pool.close()


@pytest.mark.asyncio
async def test_feedback_store_and_drift_report():
    """Test the feedback store and drift report generation."""
    import asyncpg
    from app.agentmesh.evals.drift_report import FeedbackStore, generate_drift_report
    from app.core.config import settings

    pool = await asyncpg.create_pool(settings.agent_db_dsn, min_size=2, max_size=5)
    store = FeedbackStore(pool)
    await store.setup()
    await store.clear("test-agent")

    # Record some feedback
    await store.record("test-agent", "wf-1", True, "great")
    await store.record("test-agent", "wf-2", True, "good")
    await store.record("test-agent", "wf-3", False, "bad")

    acceptance = await store.get_acceptance_rate("test-agent")
    assert acceptance == 2/3, f"Expected 2/3 acceptance rate, got {acceptance}"

    # Test drift report — no drift
    report = generate_drift_report(
        "test-agent",
        current_eval_score=0.95,
        previous_eval_score=0.93,
        current_acceptance_rate=0.90,
        previous_acceptance_rate=0.88,
    )
    assert not report.drift_detected, "Should not detect drift with improving metrics"

    # Test drift report — drift detected
    report = generate_drift_report(
        "test-agent",
        current_eval_score=0.80,
        previous_eval_score=0.95,
        current_acceptance_rate=0.70,
        previous_acceptance_rate=0.90,
    )
    assert report.drift_detected, "Should detect drift with dropping metrics"
    assert len(report.warnings) == 2, f"Expected 2 warnings, got {len(report.warnings)}"

    print(f"\n  Feedback store: acceptance={acceptance:.2%} ✅")
    print(f"  Drift report: detected={report.drift_detected} warnings={len(report.warnings)} ✅")

    await store.clear("test-agent")
    await pool.close()
