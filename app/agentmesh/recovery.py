"""Recovery record + failure taxonomy (task_3 + task_7 phase 4).

When a workflow escalates or fails, it emits a structured RecoveryRecord
instead of just a status string. The record contains everything a human
needs to understand what happened and decide what to do next — without
reconstructing from logs.

Failure taxonomy (from the harness engineering framework):
  MISSING_CONTEXT      — agent lacked info → update context map
  BAD_TOOL_CONTRACT    — schema/tool mismatch → improve spec
  MISSING_GUARDRAIL     — unsafe action attempted → add policy
  WEAK_VERIFICATION     — check missed a case → add regression test
  SIDE_EFFECT_UNKNOWN  — verification returned UNKNOWN → escalate w/ record
  MODEL_ERROR          — LLM call failed → retry/fallback path
  CONTRACT_UNMET       — done_when conditions failed → check evidence
  BUDGET_EXCEEDED      — cost budget hit → review agent behavior
  TIMEOUT              — human approval timed out → follow up
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel


class FailureType(str, Enum):
    MISSING_CONTEXT = "missing_context"
    BAD_TOOL_CONTRACT = "bad_tool_contract"
    MISSING_GUARDRAIL = "missing_guardrail"
    WEAK_VERIFICATION = "weak_verification"
    SIDE_EFFECT_UNKNOWN = "side_effect_unknown"
    MODEL_ERROR = "model_error"
    CONTRACT_UNMET = "contract_unmet"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    VERIFICATION_FAILED = "verification_failed"
    UNKNOWN = "unknown"


class RecoveryRecord(BaseModel):
    """Structured failure evidence bundle — emitted on escalation or failure.

    Instead of a status string, the workflow returns this record so a human
    can understand what happened without reconstructing from logs. The record
    contains the failure type, the state at failure, verification evidence,
    cost, and a repair hint.
    """
    run_id: str = ""
    failure_type: FailureType = FailureType.UNKNOWN
    status: str = ""  # the workflow status that triggered the record
    state_snapshot: dict[str, Any] = {}
    verification_results: dict[str, dict] = {}
    cost_incurred: float = 0.0
    cost_budget: float = 0.0
    repair_hint: str = ""
    escalation_reason: str = ""


def classify_failure(status: str, verification_results: dict[str, dict] | None = None) -> FailureType:
    """Classify a workflow failure status into a FailureType.

    This maps the workflow's status string to the failure taxonomy, using
    verification evidence to distinguish UNKNOWN from FAILED.
    """
    evidence = verification_results or {}

    if status == "verification_unknown":
        return FailureType.SIDE_EFFECT_UNKNOWN
    if status == "verification_failed":
        return FailureType.VERIFICATION_FAILED
    if status == "contract_unmet":
        return FailureType.CONTRACT_UNMET
    if status == "cost_exceeded":
        return FailureType.BUDGET_EXCEEDED
    if status == "timeout":
        return FailureType.TIMEOUT
    if status == "escalated":
        # Check if escalation was due to verification unknown
        for ev in evidence.values():
            if ev.get("status") == "unknown":
                return FailureType.SIDE_EFFECT_UNKNOWN
        return FailureType.UNKNOWN
    if status == "failed":
        return FailureType.MODEL_ERROR
    return FailureType.UNKNOWN


def build_recovery_record(
    run_id: str,
    status: str,
    state: dict | None = None,
    verification_results: dict[str, dict] | None = None,
    cost_incurred: float = 0.0,
    cost_budget: float = 0.0,
    escalation_reason: str = "",
) -> RecoveryRecord:
    """Build a RecoveryRecord from workflow failure context."""
    failure_type = classify_failure(status, verification_results)

    repair_hints = {
        FailureType.SIDE_EFFECT_UNKNOWN: "Check the external system directly — the side effect may have occurred but verification couldn't confirm it. Do NOT retry blindly.",
        FailureType.VERIFICATION_FAILED: "The side effect did not produce the expected result. Check the verification mismatches for field-level details.",
        FailureType.CONTRACT_UNMET: "One or more done_when conditions failed. Review the failed checks and their evidence.",
        FailureType.BUDGET_EXCEEDED: "The agent exceeded its cost budget. Review the agent's tool call patterns and LLM usage.",
        FailureType.TIMEOUT: "Human approval was not received within the timeout window. Follow up with the approver.",
        FailureType.MODEL_ERROR: "The LLM or graph failed. Check the workflow event history for the specific error.",
    }

    return RecoveryRecord(
        run_id=run_id,
        failure_type=failure_type,
        status=status,
        state_snapshot=state or {},
        verification_results=verification_results or {},
        cost_incurred=cost_incurred,
        cost_budget=cost_budget,
        repair_hint=repair_hints.get(failure_type, "Review the workflow event history for details."),
        escalation_reason=escalation_reason,
    )
