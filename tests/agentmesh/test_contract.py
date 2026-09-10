"""Tests for the task contract module (task_7 phase 2)."""

import pytest

from app.agentmesh.contract import (
    TaskContract,
    DoneCondition,
    EscalateCondition,
    evaluate_contract,
)


def test_empty_contract_passes():
    """No done_when conditions = no checks = passes."""
    result = evaluate_contract(TaskContract(), state={}, verification_evidence={})
    assert result.all_passed is True
    assert result.failed == []
    assert result.escalated is False


def test_done_when_suppliers_found_pass():
    """suppliers_found check passes when suppliers exist."""
    result = evaluate_contract(
        TaskContract(done_when=[DoneCondition(check="suppliers_found")]),
        state={"suppliers": [{"name": "Acme"}]},
        verification_evidence={},
    )
    assert result.all_passed is True


def test_done_when_suppliers_found_fails():
    """suppliers_found check fails when no suppliers."""
    result = evaluate_contract(
        TaskContract(done_when=[DoneCondition(check="suppliers_found")]),
        state={"suppliers": []},
        verification_evidence={},
    )
    assert result.all_passed is False
    assert "suppliers_found" in result.failed


def test_done_when_po_created_pass():
    """po_created passes when verification evidence says verified."""
    result = evaluate_contract(
        TaskContract(done_when=[DoneCondition(check="po_created")]),
        state={},
        verification_evidence={"po_created": {"status": "verified"}},
    )
    assert result.all_passed is True


def test_done_when_po_created_fails():
    """po_created fails when verification says failed."""
    result = evaluate_contract(
        TaskContract(done_when=[DoneCondition(check="po_created")]),
        state={},
        verification_evidence={"po_created": {"status": "failed"}},
    )
    assert result.all_passed is False
    assert "po_created" in result.failed


def test_escalate_budget_exceeded():
    """Escalation triggers when cost exceeds budget."""
    result = evaluate_contract(
        TaskContract(escalate_when=[EscalateCondition(trigger="budget_exceeded")]),
        state={},
        verification_evidence={},
        cost_incurred=150.0,
        cost_budget=100.0,
    )
    assert result.escalated is True
    assert "budget_exceeded" in result.escalation_reason


def test_escalate_repeated_rejection():
    """Escalation triggers when rejection count hits threshold."""
    result = evaluate_contract(
        TaskContract(escalate_when=[EscalateCondition(trigger="repeated_rejection", max_occurrences=3)]),
        state={"rejection_count": 3},
        verification_evidence={},
    )
    assert result.escalated is True
    assert "repeated_rejection" in result.escalation_reason


def test_escalate_verification_unknown():
    """Escalation triggers when any verification returns UNKNOWN."""
    result = evaluate_contract(
        TaskContract(escalate_when=[EscalateCondition(trigger="verification_unknown")]),
        state={},
        verification_evidence={"po_created": {"status": "unknown", "error": "db_query_failed"}},
    )
    assert result.escalated is True
    assert "verification_unknown" in result.escalation_reason


def test_escalation_overrides_done_when():
    """If escalated, done_when is not evaluated."""
    result = evaluate_contract(
        TaskContract(
            done_when=[DoneCondition(check="suppliers_found")],
            escalate_when=[EscalateCondition(trigger="budget_exceeded")],
        ),
        state={"suppliers": []},  # would fail done_when
        verification_evidence={},
        cost_incurred=200.0,
        cost_budget=100.0,
    )
    assert result.escalated is True
    assert result.failed == []  # done_when not evaluated


def test_unknown_check_fails_safe():
    """An unknown check name fails safe (does not pass)."""
    result = evaluate_contract(
        TaskContract(done_when=[DoneCondition(check="nonexistent_check")]),
        state={},
        verification_evidence={},
    )
    assert result.all_passed is False
    assert "nonexistent_check" in result.failed


def test_optional_condition_not_required():
    """A non-required condition doesn't block completion even if it fails."""
    result = evaluate_contract(
        TaskContract(done_when=[DoneCondition(check="suppliers_found", required=False)]),
        state={"suppliers": []},
        verification_evidence={},
    )
    assert result.all_passed is True
