"""Generic eval harness — CI-gated eval runner.

Week 6: extended with:
  - RAG retrieval metrics (recall@k, MRR)
  - LLM-as-judge scoring of the Decide node's rationale
  - Semantic cache hit rate tracking
  - CI gate: fails the build if any score regresses below threshold

Agent-agnostic: the harness never imports from agents/. The caller
provides the dataset and the invoke_agent callback.
"""

import logging
from collections.abc import Awaitable, Callable

from app.agentmesh.evals.models import EvalResult, EvalScenario, EvalScorecard
from app.core.config import settings

logger = logging.getLogger("agentmesh.evals")


class EvalHarness:
    """Generic eval harness — scores tool-selection, outcome, RAG, and judge quality.

    Agent-agnostic: the harness never imports from agents/. The caller
    provides the dataset and the invoke_agent callback.
    """

    async def run(
        self,
        dataset: list[EvalScenario],
        invoke_agent: Callable[[dict], Awaitable[dict]],
    ) -> list[EvalResult]:
        """Run all scenarios, compare expected vs actual, return scored results."""
        results = []
        for scenario in dataset:
            try:
                actual = await invoke_agent(scenario.input)
            except Exception as e:
                results.append(
                    EvalResult(
                        scenario_id=scenario.scenario_id,
                        passed=False,
                        tool_accuracy=0.0,
                        outcome_match=False,
                        actual_tool_sequence=[],
                        actual_outcome={},
                        errors=[f"Agent raised: {e}"],
                    )
                )
                continue

            actual_tools = actual.get("tool_calls", [])
            tool_accuracy = self._score_tool_sequence(
                scenario.expected_tool_sequence, actual_tools
            )
            outcome_match = self._score_outcome(
                scenario.expected_outcome, actual
            )

            # Week 6: RAG retrieval metrics
            rag_recall = None
            rag_mrr = None
            if scenario.expected_retrieval_ids and actual.get("retrieval_results"):
                rag_recall = self._score_rag_recall(
                    scenario.expected_retrieval_ids,
                    actual["retrieval_results"],
                    k=5,
                )
                rag_mrr = self._score_rag_mrr(
                    scenario.expected_retrieval_ids,
                    actual["retrieval_results"],
                )

            # Week 6: LLM-as-judge score
            judge_score = None
            judge_rationale = None
            if actual.get("decision_rationale"):
                judge_score, judge_rationale = self._llm_as_judge(
                    scenario, actual
                )

            passed = tool_accuracy == 1.0 and outcome_match
            if rag_recall is not None and rag_recall < settings.eval_min_rag_recall:
                passed = False

            results.append(
                EvalResult(
                    scenario_id=scenario.scenario_id,
                    passed=passed,
                    tool_accuracy=tool_accuracy,
                    outcome_match=outcome_match,
                    actual_tool_sequence=actual_tools,
                    actual_outcome=actual,
                    errors=(
                        []
                        if passed
                        else self._collect_errors(scenario, actual_tools, actual)
                    ),
                    rag_recall_at_k=rag_recall,
                    rag_mrr=rag_mrr,
                    judge_score=judge_score,
                    judge_rationale=judge_rationale,
                )
            )
        return results

    def scorecard(
        self,
        results: list[EvalResult],
        semantic_cache_hits: int = 0,
        semantic_cache_total: int = 0,
    ) -> EvalScorecard:
        """Compute aggregate scorecard from results."""
        total = len(results)
        if total == 0:
            return EvalScorecard()

        passed = sum(1 for r in results if r.passed)
        avg_tool = sum(r.tool_accuracy for r in results) / total
        completion = sum(1 for r in results if r.outcome_match) / total

        rag_results = [r for r in results if r.rag_recall_at_k is not None]
        avg_rag = (
            sum(r.rag_recall_at_k for r in rag_results) / len(rag_results)
            if rag_results
            else None
        )

        judge_results = [r for r in results if r.judge_score is not None]
        avg_judge = (
            sum(r.judge_score for r in judge_results) / len(judge_results)
            if judge_results
            else None
        )

        cache_hit_rate = (
            semantic_cache_hits / semantic_cache_total
            if semantic_cache_total > 0
            else None
        )

        return EvalScorecard(
            total_scenarios=total,
            passed=passed,
            failed=total - passed,
            avg_tool_accuracy=avg_tool,
            completion_rate=completion,
            avg_rag_recall=avg_rag,
            avg_judge_score=avg_judge,
            semantic_cache_hit_rate=cache_hit_rate,
        )

    def check_ci_gate(self, scorecard: EvalScorecard) -> tuple[bool, list[str]]:
        """Check if the scorecard passes the CI gate.

        Returns (passed, failures) where failures is a list of threshold
        violations.
        """
        failures = []

        if scorecard.avg_tool_accuracy < settings.eval_min_tool_accuracy:
            failures.append(
                f"Tool accuracy {scorecard.avg_tool_accuracy:.2%} < "
                f"threshold {settings.eval_min_tool_accuracy:.2%}"
            )

        if scorecard.completion_rate < settings.eval_min_completion_rate:
            failures.append(
                f"Completion rate {scorecard.completion_rate:.2%} < "
                f"threshold {settings.eval_min_completion_rate:.2%}"
            )

        if (
            scorecard.avg_rag_recall is not None
            and scorecard.avg_rag_recall < settings.eval_min_rag_recall
        ):
            failures.append(
                f"RAG recall {scorecard.avg_rag_recall:.2%} < "
                f"threshold {settings.eval_min_rag_recall:.2%}"
            )

        return (len(failures) == 0, failures)

    def _score_tool_sequence(
        self, expected: list[str], actual: list[str]
    ) -> float:
        """Score tool sequence accuracy.

        Full match = 1.0. Partial match = fraction of correct tools in order.
        """
        if not expected:
            return 1.0 if not actual else 0.0
        correct = 0
        for i, tool in enumerate(expected):
            if i < len(actual) and actual[i] == tool:
                correct += 1
        return correct / len(expected)

    def _score_outcome(self, expected: dict, actual: dict) -> bool:
        """Check if the outcome matches expectations."""
        for key, value in expected.items():
            if actual.get(key) != value:
                return False
        return True

    def _score_rag_recall(
        self, expected_ids: list[str], actual_results: list[dict], k: int = 5
    ) -> float:
        """Recall@k: fraction of expected IDs in the top-k actual results."""
        actual_ids = [r.get("id", "") for r in actual_results[:k]]
        hits = sum(1 for eid in expected_ids if eid in actual_ids)
        return hits / len(expected_ids) if expected_ids else 1.0

    def _score_rag_mrr(
        self, expected_ids: list[str], actual_results: list[dict]
    ) -> float:
        """Mean Reciprocal Rank: 1/rank of the first relevant result."""
        for i, result in enumerate(actual_results, 1):
            if result.get("id", "") in expected_ids:
                return 1.0 / i
        return 0.0

    def _llm_as_judge(
        self, scenario: EvalScenario, actual: dict
    ) -> tuple[float, str]:
        """LLM-as-judge scoring of the agent's decision rationale.

        In production, this would call an LLM with a rubric to score
        the agent's rationale. For self-contained demo, we use a
        rule-based judge that checks if the rationale mentions the
        key factors (price, rating, lead time).

        Returns (score 0.0-1.0, rationale_string).
        """
        rationale = actual.get("decision_rationale", "")
        if not rationale:
            return 0.0, "No rationale provided"

        # Check for key decision factors in the rationale
        factors = ["price", "rating", "lead", "supplier", "selected"]
        mentioned = sum(1 for f in factors if f.lower() in rationale.lower())
        score = min(mentioned / 3.0, 1.0)  # 3+ factors = full score

        rubric = scenario.judge_rubric or "Rationale should mention price, rating, and lead time."
        judge_rationale = (
            f"Score {score:.2f}: rationale mentions {mentioned}/{len(factors)} key factors. "
            f"Rubric: {rubric}"
        )

        return score, judge_rationale

    def _collect_errors(
        self,
        scenario: EvalScenario,
        actual_tools: list[str],
        actual: dict,
    ) -> list[str]:
        errors = []
        if actual_tools != scenario.expected_tool_sequence:
            errors.append(
                f"Tool sequence mismatch: expected {scenario.expected_tool_sequence}, got {actual_tools}"
            )
        for key, value in scenario.expected_outcome.items():
            if actual.get(key) != value:
                errors.append(
                    f"Outcome mismatch for '{key}': expected {value}, got {actual.get(key)}"
                )
        return errors


# Singleton
harness = EvalHarness()
