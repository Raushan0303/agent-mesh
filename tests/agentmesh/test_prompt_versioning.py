"""Tests for prompt versioning — hash determinism, change detection, state storage.

These tests verify that:
1. The same prompt always produces the same hash (determinism)
2. Changing a single character in the prompt changes the hash
3. The hash is stored in the graph state by the decide node
4. The version tag is human-readable and matches the constant
5. LLMMessage dataclass instances and dicts produce the same hash
"""

import pytest

from app.agentmesh.prompt_versioning import PromptVersion, hash_prompt, version_prompt
from app.agentmesh.llm import LLMMessage


# ── hash_prompt tests ──


def test_hash_is_deterministic():
    """Same messages → same hash, every time."""
    messages = [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": "Hello"},
    ]
    h1 = hash_prompt(messages)
    h2 = hash_prompt(messages)
    assert h1 == h2
    assert len(h1) == 16  # SHA256[:16]


def test_hash_changes_on_content_change():
    """Changing a single character in the prompt changes the hash."""
    messages_a = [{"role": "user", "content": "Select supplier A."}]
    messages_b = [{"role": "user", "content": "Select supplier B."}]
    assert hash_prompt(messages_a) != hash_prompt(messages_b)


def test_hash_changes_on_role_change():
    """Changing the role (system vs user) changes the hash."""
    messages_a = [{"role": "system", "content": "You are an agent."}]
    messages_b = [{"role": "user", "content": "You are an agent."}]
    assert hash_prompt(messages_a) != hash_prompt(messages_b)


def test_hash_independent_of_list_order():
    """Reordering messages changes the hash (order matters in prompts)."""
    messages_a = [
        {"role": "system", "content": "System msg"},
        {"role": "user", "content": "User msg"},
    ]
    messages_b = [
        {"role": "user", "content": "User msg"},
        {"role": "system", "content": "System msg"},
    ]
    assert hash_prompt(messages_a) != hash_prompt(messages_b)


def test_hash_accepts_llm_message_dataclass():
    """LLMMessage dataclass instances work the same as dicts."""
    messages_dict = [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": "Hello"},
    ]
    messages_dataclass = [
        LLMMessage(role="system", content="You are a test agent."),
        LLMMessage(role="user", content="Hello"),
    ]
    assert hash_prompt(messages_dict) == hash_prompt(messages_dataclass)


def test_hash_is_16_chars():
    """Hash is truncated to 16 characters for readability."""
    messages = [{"role": "user", "content": "test"}]
    h = hash_prompt(messages)
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


# ── version_prompt tests ──


def test_version_prompt_returns_prompt_version():
    """version_prompt returns a PromptVersion with messages, hash, and version."""
    messages = [
        LLMMessage(role="system", content="Test"),
        LLMMessage(role="user", content="Hello"),
    ]
    tagged = version_prompt(messages, version="v1")
    assert isinstance(tagged, PromptVersion)
    assert tagged.messages == messages
    assert tagged.hash == hash_prompt(messages)
    assert tagged.version == "v1"


def test_version_prompt_is_frozen():
    """PromptVersion is immutable — can't accidentally mutate it."""
    messages = [LLMMessage(role="user", content="test")]
    tagged = version_prompt(messages, version="v1")
    with pytest.raises(AttributeError):
        tagged.version = "v2"


# ── Sourcing agent prompt tests ──


def test_sourcing_decide_prompt_has_hash_and_version():
    """build_decide_prompt returns a PromptVersion, not a bare list."""
    from app.agents.sourcing_agent.prompts import build_decide_prompt, DECIDE_PROMPT_VERSION

    scored = [
        {"name": "Supplier A", "price": 2.50, "rating": 4.5, "lead_time_days": 7},
        {"name": "Supplier B", "price": 3.00, "rating": 4.8, "lead_time_days": 14},
    ]
    result = build_decide_prompt("USB-C cable", 500, 3.00, scored, [])

    assert isinstance(result, PromptVersion)
    assert result.version == DECIDE_PROMPT_VERSION
    assert len(result.hash) == 16
    assert len(result.messages) == 2  # system + user


def test_sourcing_decide_prompt_hash_changes_with_input():
    """Different supplier data → different hash (the prompt content changed)."""
    from app.agents.sourcing_agent.prompts import build_decide_prompt

    scored_a = [{"name": "Supplier A", "price": 2.50, "rating": 4.5, "lead_time_days": 7}]
    scored_b = [{"name": "Supplier B", "price": 3.00, "rating": 4.8, "lead_time_days": 14}]

    result_a = build_decide_prompt("USB-C cable", 500, 3.00, scored_a, [])
    result_b = build_decide_prompt("USB-C cable", 500, 3.00, scored_b, [])
    assert result_a.hash != result_b.hash


def test_sourcing_decide_prompt_hash_deterministic():
    """Same inputs → same hash, every time."""
    from app.agents.sourcing_agent.prompts import build_decide_prompt

    scored = [{"name": "Supplier A", "price": 2.50, "rating": 4.5, "lead_time_days": 7}]
    r1 = build_decide_prompt("USB-C cable", 500, 3.00, scored, [])
    r2 = build_decide_prompt("USB-C cable", 500, 3.00, scored, [])
    assert r1.hash == r2.hash


# ── Hiring agent prompt tests ──


def test_hiring_screen_prompt_has_hash_and_version():
    """_build_screen_prompt returns a PromptVersion."""
    from app.agents.hiring_agent.graph import _build_screen_prompt, SCREEN_PROMPT_VERSION

    result = _build_screen_prompt("Senior Backend Engineer", "Python, FastAPI, Postgres", "")
    assert isinstance(result, PromptVersion)
    assert result.version == SCREEN_PROMPT_VERSION
    assert len(result.hash) == 16


def test_hiring_interview_prompt_has_hash_and_version():
    """_build_interview_prompt returns a PromptVersion."""
    from app.agents.hiring_agent.graph import (
        _build_interview_prompt,
        INTERVIEW_PROMPT_VERSION,
    )
    from app.agents.hiring_agent.state import HiringBriefInput

    brief = HiringBriefInput(
        candidate_name="Test Candidate",
        role="Backend Engineer",
        resume_text="Python developer",
        years_experience=5,
        target_salary=120000,
    )
    result = _build_interview_prompt(brief, ["Python"], 0, [])
    assert isinstance(result, PromptVersion)
    assert result.version == INTERVIEW_PROMPT_VERSION
    assert len(result.hash) == 16


def test_hiring_offer_prompt_has_hash_and_version():
    """_build_offer_prompt returns a PromptVersion."""
    from app.agents.hiring_agent.graph import _build_offer_prompt, OFFER_PROMPT_VERSION
    from app.agents.hiring_agent.state import HiringBriefInput

    brief = HiringBriefInput(
        candidate_name="Test Candidate",
        role="Backend Engineer",
        resume_text="Python developer",
        years_experience=5,
        target_salary=120000,
    )
    result = _build_offer_prompt(brief, 85.0)
    assert isinstance(result, PromptVersion)
    assert result.version == OFFER_PROMPT_VERSION
    assert len(result.hash) == 16


# ── State field tests ──


def test_sourcing_state_has_prompt_fields():
    """AgentState includes prompt_hash and prompt_version fields."""
    from app.agents.sourcing_agent.state import AgentState

    # TypedDict fields are accessible as annotations
    annotations = AgentState.__annotations__
    assert "prompt_hash" in annotations
    assert "prompt_version" in annotations


def test_hiring_state_has_prompt_fields():
    """Hiring AgentState includes prompt hash/version fields for all 3 LLM nodes."""
    from app.agents.hiring_agent.state import AgentState

    annotations = AgentState.__annotations__
    assert "screen_prompt_hash" in annotations
    assert "screen_prompt_version" in annotations
    assert "interview_prompt_hash" in annotations
    assert "interview_prompt_version" in annotations
    assert "offer_prompt_hash" in annotations
    assert "offer_prompt_version" in annotations


def test_sourcing_result_has_prompt_version():
    """SourcingResult includes prompt_version in its output schema."""
    from app.agents.sourcing_agent.state import SourcingResult

    fields = SourcingResult.model_fields
    assert "prompt_version" in fields
