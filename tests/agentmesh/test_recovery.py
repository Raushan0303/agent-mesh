"""Tests for the recovery record + failure taxonomy (task_3 + task_7 phase 4)."""

from app.agentmesh.recovery import (
    FailureType,
    RecoveryRecord,
    classify_failure,
    build_recovery_record,
)


def test_classify_verification_unknown():
    assert classify_failure("verification_unknown") == FailureType.SIDE_EFFECT_UNKNOWN


def test_classify_verification_failed():
    assert classify_failure("verification_failed") == FailureType.VERIFICATION_FAILED


def test_classify_contract_unmet():
    assert classify_failure("contract_unmet") == FailureType.CONTRACT_UNMET


def test_classify_cost_exceeded():
    assert classify_failure("cost_exceeded") == FailureType.BUDGET_EXCEEDED


def test_classify_timeout():
    assert classify_failure("timeout") == FailureType.TIMEOUT


def test_classify_escalated_with_unknown_evidence():
    evidence = {"po_created": {"status": "unknown"}}
    assert classify_failure("escalated", evidence) == FailureType.SIDE_EFFECT_UNKNOWN


def test_classify_escalated_without_unknown_evidence():
    assert classify_failure("escalated") == FailureType.UNKNOWN


def test_build_recovery_record_side_effect_unknown():
    record = build_recovery_record(
        run_id="wf-123",
        status="verification_unknown",
        state={"suppliers": [{"name": "Acme"}]},
        verification_results={"po_created": {"status": "unknown", "error": "db_query_failed"}},
        cost_incurred=1.50,
        cost_budget=10.0,
    )
    assert record.failure_type == FailureType.SIDE_EFFECT_UNKNOWN
    assert record.run_id == "wf-123"
    assert "Do NOT retry" in record.repair_hint
    assert record.cost_incurred == 1.50


def test_build_recovery_record_budget_exceeded():
    record = build_recovery_record(
        run_id="wf-456",
        status="cost_exceeded",
        cost_incurred=15.0,
        cost_budget=10.0,
    )
    assert record.failure_type == FailureType.BUDGET_EXCEEDED
    assert "cost budget" in record.repair_hint.lower()


def test_build_recovery_record_timeout():
    record = build_recovery_record(
        run_id="wf-789",
        status="timeout",
    )
    assert record.failure_type == FailureType.TIMEOUT
    assert "follow up" in record.repair_hint.lower()


def test_recovery_record_is_serializable():
    """RecoveryRecord must be JSON-serializable for the API response."""
    record = build_recovery_record(
        run_id="wf-serial",
        status="verification_unknown",
        state={"key": "value"},
        verification_results={"check": {"status": "unknown"}},
    )
    dumped = record.model_dump()
    assert isinstance(dumped, dict)
    assert dumped["failure_type"] == "side_effect_unknown"
    assert dumped["run_id"] == "wf-serial"
