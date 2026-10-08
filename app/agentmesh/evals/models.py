"""Eval scenario and result models — agent-agnostic schema.

Week 6: extended with RAG retrieval metrics and LLM-as-judge fields.
"""

from dataclasses import dataclass, field


@dataclass
class EvalScenario:
    """A single eval scenario — agent-agnostic schema.

    The platform's eval harness operates on this type. The agent
    provides the dataset; the harness scores it.
    """

    scenario_id: str
    input: dict
    expected_tool_sequence: list[str]
    expected_outcome: dict
    # Week 6: optional RAG retrieval ground truth
    expected_retrieval_query: str | None = None
    expected_retrieval_ids: list[str] | None = None
    # Week 6: optional LLM-as-judge rubric for the Decide rationale
    judge_rubric: str | None = None


@dataclass
class EvalResult:
    """Result of running one eval scenario."""

    scenario_id: str
    passed: bool
    tool_accuracy: float
    outcome_match: bool
    actual_tool_sequence: list[str]
    actual_outcome: dict
    errors: list[str] = field(default_factory=list)
    # Week 6: RAG retrieval metrics
    rag_recall_at_k: float | None = None
    rag_mrr: float | None = None
    # Week 6: LLM-as-judge score (0.0 to 1.0)
    judge_score: float | None = None
    judge_rationale: str | None = None


@dataclass
class EvalScorecard:
    """Aggregate scorecard across all scenarios in a dataset."""

    total_scenarios: int = 0
    passed: int = 0
    failed: int = 0
    avg_tool_accuracy: float = 0.0
    completion_rate: float = 0.0
    avg_rag_recall: float | None = None
    avg_judge_score: float | None = None
    semantic_cache_hit_rate: float | None = None
    planner: str = ""  # "llm" | "fixed" — fixed means tool accuracy is by construction

    @property
    def pass_rate(self) -> float:
        if self.total_scenarios == 0:
            return 0.0
        return self.passed / self.total_scenarios

    def to_dict(self) -> dict:
        return {
            "total_scenarios": self.total_scenarios,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
            "avg_tool_accuracy": round(self.avg_tool_accuracy, 4),
            "completion_rate": round(self.completion_rate, 4),
            "avg_rag_recall": round(self.avg_rag_recall, 4) if self.avg_rag_recall is not None else None,
            "avg_judge_score": round(self.avg_judge_score, 4) if self.avg_judge_score is not None else None,
            "semantic_cache_hit_rate": round(self.semantic_cache_hit_rate, 4) if self.semantic_cache_hit_rate is not None else None,
        }
