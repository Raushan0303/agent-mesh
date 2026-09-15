"""Tests for supply chain + exfiltration + identity defenses (task_6)."""

from app.agentmesh.supply_chain_defense import (
    inspect_payload,
    PayloadVerdict,
    ToolScope,
    ToolScopeRegistry,
    tool_scope_registry,
)


def test_clean_payload_passes():
    result = inspect_payload("Supplier: Acme Corp, price: $2.50")
    assert result.verdict == PayloadVerdict.CLEAN
    assert result.patterns_matched == []


def test_api_key_detected_and_redacted():
    result = inspect_payload("Result: sk-abc123def456ghi789jkl012mno345pqr678")
    assert result.verdict == PayloadVerdict.REDACTED
    assert "api_key_openai" in result.patterns_matched
    assert "[REDACTED:api_key_openai]" in result.inspected_payload
    assert "sk-abc123" not in result.inspected_payload


def test_credit_card_detected():
    result = inspect_payload("Card: 4111-1111-1111-1111")
    assert "credit_card_number" in result.patterns_matched
    assert result.verdict == PayloadVerdict.REDACTED


def test_ssn_detected():
    result = inspect_payload("SSN: 123-45-6789")
    assert "ssn" in result.patterns_matched


def test_email_detected():
    result = inspect_payload("Contact: alice@acme.com")
    assert "email_address" in result.patterns_matched


def test_bearer_token_detected():
    result = inspect_payload("Auth: Bearer eyJhbGciOiJIUzI1")
    assert "bearer_token" in result.patterns_matched


def test_private_key_detected():
    result = inspect_payload("-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...")
    assert "private_key" in result.patterns_matched


def test_no_redact_when_disabled():
    result = inspect_payload("sk-abc123def456ghi789jkl012mno345pqr678", redact=False)
    assert result.verdict == PayloadVerdict.FLAGGED
    assert "sk-abc123" in result.inspected_payload  # not redacted


def test_dict_payload_scanned():
    result = inspect_payload({"data": "sk-abc123def456ghi789jkl012mno345pqr678"})
    assert "api_key_openai" in result.patterns_matched


def test_empty_payload_passes():
    result = inspect_payload("")
    assert result.verdict == PayloadVerdict.CLEAN


def test_tool_scope_allows_allowed_tool():
    scope = ToolScope(agent_name="sourcing_agent", allowed_tools={"query_suppliers"})
    assert scope.can_call("query_suppliers") is True
    assert scope.can_call("initiate_payment") is False


def test_tool_scope_registry_check_access():
    reg = ToolScopeRegistry()
    reg.register_scope("sourcing_agent", ["query_suppliers", "create_purchase_order"])

    assert reg.check_access("sourcing_agent", "query_suppliers") is True
    assert reg.check_access("sourcing_agent", "create_purchase_order") is True
    assert reg.check_access("sourcing_agent", "send_offer") is False


def test_tool_scope_registry_no_scope_denies():
    reg = ToolScopeRegistry()
    assert reg.check_access("unknown_agent", "any_tool") is False


def test_tool_scope_add_remove():
    scope = ToolScope(agent_name="test", allowed_tools={"tool_a"})
    scope.add_tool("tool_b")
    assert scope.can_call("tool_b") is True
    scope.remove_tool("tool_a")
    assert scope.can_call("tool_a") is False
