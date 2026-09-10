"""Task contract — bounded task object with done_when and escalate_when (task_7 phase 2).

Turns "the agent finished" into "the agent proved the contract." Without
a contract, an agent can solve an easier version of the problem and
confidently declare success. The contract gives the harness something
objective to evaluate.

done_when: deterministic checks that MUST pass before status="completed"
escalate_when: conditions that force human escalation (not LLM judgment)

Key property: done_when conditions are evaluated against EVIDENCE
(verification results, state fields), not against the LLM's self-report.
"po_created" means verify_po_exists returned VERIFIED — not "the LLM
said it created a PO."
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel


class DoneCondition(BaseModel):
    """A single objective completion check.

    check: a named check that the evaluator knows how to run:
      - "suppliers_found" — at least one supplier in state
      - "po_created" — PO verification returned VERIFIED
      - "payment_initiated" — payment verification returned VERIFIED
      - "offer_sent" — offer verification returned VERIFIED
      - "cost_within_budget" — total cost <= budget
    required: if True, failing this check blocks completion
    """
    check: str
    required: bool = True


class EscalateCondition(BaseModel):
    """A condition that forces human escalation.

    trigger: a named trigger that the evaluator knows how to check:
      - "budget_exceeded" — total cost > budget
      - "repeated_rejection" — rejection_count >= max_occurrences
      - "verification_unknown" — any verification returned UNKNOWN
      - "no_matches" — graph returned no_matches status
    max_occurrences: for "repeated_rejection", the threshold
    """
    trigger: str
    max_occurrences: int = 1


class TaskContract(BaseModel):
    """The bounded task object — what the agent must prove to declare done."""
    constraints: list[str] = []
    deliverable: str = ""
    done_when: list[DoneCondition] = []
    escalate_when: list[EscalateCondition] = []


class ContractResult(BaseModel):
    """Result of evaluating a contract against state + verification evidence."""
    all_passed: bool
    failed: list[str] = []
    escalated: bool = False
    escalation_reason: str = ""


def evaluate_contract(
    contract: TaskContract,
    state: dict,
    verification_evidence: dict[str, dict] | None = None,
    cost_incurred: float = 0.0,
    cost_budget: float = 0.0,
) -> ContractResult:
    """Evaluate done_when and escalate_when against evidence.

    Args:
        contract: the task contract
        state: the graph state (suppliers, status, rejection_count, etc.)
        verification_evidence: map of check_name → verification result dict
            e.g. {"po_created": {"status": "verified"}, ...}
        cost_incurred: total cost so far
        cost_budget: the budget limit

    Returns:
        ContractResult with all_passed, failed checks, and escalation status
    """
    evidence = verification_evidence or {}
    failed: list[str] = []
    escalated = False
    escalation_reason = ""

    # Check escalate_when first — escalation overrides done_when
    for esc in contract.escalate_when:
        if esc.trigger == "budget_exceeded" and cost_budget > 0 and cost_incurred > cost_budget:
            escalated = True
            escalation_reason = f"budget_exceeded: spent={cost_incurred} budget={cost_budget}"
            break
        if esc.trigger == "repeated_rejection":
            if state.get("rejection_count", 0) >= esc.max_occurrences:
                escalated = True
                escalation_reason = f"repeated_rejection: {state.get('rejection_count', 0)} >= {esc.max_occurrences}"
                break
        if esc.trigger == "verification_unknown":
            for ev in evidence.values():
                if ev.get("status") == "unknown":
                    escalated = True
                    escalation_reason = f"verification_unknown: {ev.get('error', '')}"
                    break
            if escalated:
                break
        if esc.trigger == "no_matches" and state.get("status") == "no_matches":
            escalated = True
            escalation_reason = "no_matches: graph found no suppliers"
            break

    if escalated:
        return ContractResult(all_passed=False, failed=[], escalated=True, escalation_reason=escalation_reason)

    # Check done_when — all required conditions must pass
    for cond in contract.done_when:
        if not cond.required:
            continue
        passed = _evaluate_check(cond.check, state, evidence, cost_incurred, cost_budget)
        if not passed:
            failed.append(cond.check)

    return ContractResult(
        all_passed=len(failed) == 0,
        failed=failed,
        escalated=False,
    )


def _evaluate_check(
    check: str,
    state: dict,
    evidence: dict[str, dict],
    cost_incurred: float,
    cost_budget: float,
) -> bool:
    """Evaluate a single done_when check against evidence."""
    if check == "suppliers_found":
        return len(state.get("suppliers", [])) > 0
    if check == "po_created":
        return evidence.get("po_created", {}).get("status") == "verified"
    if check == "payment_initiated":
        return evidence.get("payment_initiated", {}).get("status") == "verified"
    if check == "offer_sent":
        return evidence.get("offer_sent", {}).get("status") == "verified"
    if check == "cost_within_budget":
        return cost_budget <= 0 or cost_incurred <= cost_budget
    # Unknown check — fail safe (don't pass what we can't evaluate)
    return False
