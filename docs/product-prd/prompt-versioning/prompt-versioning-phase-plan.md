# Prompt Versioning — Phase-wise Execution Plan

> **What this document is:** A granular, phase-by-phase execution plan for the prompt versioning feature. Each phase specifies exactly which files to create or modify, which interfaces to implement, and how to verify the phase is complete. This is the document you follow line-by-line to build the feature.

---

## Phase Overview

| Phase | What | Duration | Files |
|---|---|---|---|
| Phase 1 | Core module — `PromptVersion`, `hash_prompt()`, `version_prompt()` | Day 1 (morning) | `app/agentmesh/prompt_versioning.py` |
| Phase 2 | Sourcing agent — state fields, prompt builder, graph node, workflow | Day 1 (afternoon) | `state.py`, `prompts.py`, `graph.py`, `workflow.py` |
| Phase 3 | Hiring agent — state fields, 3 prompt builders, 3 graph nodes | Day 2 (morning) | `state.py`, `graph.py` |
| Phase 4 | Tests — hash determinism, change detection, state fields, immutability | Day 2 (afternoon) | `tests/agentmesh/test_prompt_versioning.py` |
| Phase 5 | Verification + commit | Day 3 | All files |

---

## Phase 1 — Core Module

**Goal:** Agent-agnostic prompt versioning utility — hash and tag any list of LLM messages.

**File to create:**
```
app/agentmesh/prompt_versioning.py
```

**Interface:**
```python
@dataclass(frozen=True)
class PromptVersion:
    """Immutable record of which prompt was sent to the LLM."""
    messages: list[dict[str, str]]
    hash: str       # SHA256[:16] of serialized messages
    version: str    # human-readable tag (e.g. "v1")


def hash_prompt(messages: list[dict[str, str]]) -> str:
    """Deterministic SHA256[:16] of JSON-serialized messages.

    Normalizes LLMMessage dataclass instances to dicts.
    Sorts JSON keys for ordering independence.
    Truncates to 16 chars for readability.
    """


def version_prompt(
    messages: list[Any],
    version: str,
) -> PromptVersion:
    """Tag a prompt with its hash and version.

    Args:
        messages: LLM messages (list of LLMMessage or dict).
        version: Human-readable version tag (e.g. "v1").

    Returns:
        PromptVersion with messages, hash, and version.
    """
```

**Key design decisions:**

1. **Hash the rendered prompt, not the template** — the hash includes the variables (item, quantity, suppliers). This means the same template with different inputs produces different hashes. This is correct — you want to know exactly what the LLM saw, not just which template was used.

2. **Version is manual, hash is automatic** — `prompt_version = "v1"` is set by the developer. When you change the prompt, bump it to `"v2"`. The hash is computed automatically. The version is for humans ("which prompt version?"), the hash is for machines ("is this the exact same prompt?").

3. **Frozen dataclass** — `PromptVersion` is immutable. You can't accidentally mutate the hash or version after creation. This prevents a class of bugs where the prompt version is changed after the hash is computed.

4. **LLMMessage and dict compatibility** — `hash_prompt()` normalizes `LLMMessage` dataclass instances to dicts before hashing. This means the same prompt produces the same hash regardless of whether it was built with `LLMMessage(role="system", content="...")` or `{"role": "system", "content": "..."}`.

**Verification:**
```python
from app.agentmesh.prompt_versioning import hash_prompt, version_prompt
from app.agentmesh.llm import LLMMessage

# Hash is deterministic
messages = [{"role": "user", "content": "test"}]
assert hash_prompt(messages) == hash_prompt(messages)

# Hash changes on content change
assert hash_prompt([{"role": "user", "content": "A"}]) != hash_prompt([{"role": "user", "content": "B"}])

# LLMMessage and dict produce the same hash
dict_msgs = [{"role": "system", "content": "test"}]
llm_msgs = [LLMMessage(role="system", content="test")]
assert hash_prompt(dict_msgs) == hash_prompt(llm_msgs)

# version_prompt returns PromptVersion
tagged = version_prompt(llm_msgs, version="v1")
assert tagged.hash == hash_prompt(llm_msgs)
assert tagged.version == "v1"
```

**Decoupling check:** `grep -r "sourcing\|hiring\|decide\|screen\|interview\|offer" app/agentmesh/prompt_versioning.py` must return ZERO results. The module is agent-agnostic.

---

## Phase 2 — Sourcing Agent Integration

**Goal:** The sourcing agent's decide node records `prompt_hash` and `prompt_version` in the LangGraph state.

**Files to modify:**
```
app/agents/sourcing_agent/state.py       # Add fields to AgentState + SourcingResult
app/agents/sourcing_agent/prompts.py     # Return PromptVersion instead of bare list
app/agents/sourcing_agent/graph.py       # Unpack PromptVersion, store in state
app/agents/sourcing_agent/workflow.py    # Include prompt_version in SourcingResult
```

### Step 2a — State fields (`state.py`)

Add to `AgentState` TypedDict, in the "Decide node output" section:
```python
class AgentState(TypedDict, total=False):
    # ... existing fields ...

    # Decide node output
    selected_supplier: dict
    decision_reason: str
    prompt_hash: str        # NEW — SHA256[:16] of the rendered decide prompt
    prompt_version: str     # NEW — human-readable version tag (e.g. "v1")
```

Add to `SourcingResult` Pydantic model:
```python
class SourcingResult(BaseModel):
    # ... existing fields ...
    prompt_version: str = ""  # NEW — which prompt version was used for the decide node
```

### Step 2b — Prompt builder (`prompts.py`)

Add version constant and change return type:
```python
from app.agentmesh.prompt_versioning import PromptVersion, version_prompt

# Bump this when you change the decide prompt template.
DECIDE_PROMPT_VERSION = "v1"

def build_decide_prompt(...) -> PromptVersion:
    """Build the LLM prompt for the Decide node.

    Returns a PromptVersion containing messages, hash, and version.
    """
    # ... existing prompt construction logic ...

    return version_prompt(
        [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=user),
        ],
        version=DECIDE_PROMPT_VERSION,
    )
```

### Step 2c — Graph node (`graph.py`)

Unpack the `PromptVersion` and store hash + version in the state:
```python
async def decide_node(state: AgentState) -> dict:
    # ... existing logic ...

    # Build the LLM prompt (returns PromptVersion with hash + version)
    prompt = build_decide_prompt(item, quantity, budget, scored, past_decisions)

    # Pass .messages to the LLM client
    response = await client.complete(
        messages=prompt.messages,
        temperature=0.0,
        max_tokens=1000,
    )

    # Store hash + version in the state dict
    return {
        "selected_supplier": selected,
        "decision_reason": rationale,
        "past_decisions": past_decisions,
        "cost_incurred": response.cost_usd,
        "prompt_hash": prompt.hash,         # NEW
        "prompt_version": prompt.version,   # NEW
    }
```

Also update the log line to include `prompt_hash` and `prompt_version`:
```python
logger.info(
    "DECIDE_LLM_COMPLETED selected=%s model=%s tokens=%s cost_usd=%.6f prompt_hash=%s prompt_version=%s",
    selected["name"], response.model, response.usage.get("total_tokens", "?"),
    response.cost_usd, prompt.hash, prompt.version,
)
```

### Step 2d — Workflow (`workflow.py`)

Include `prompt_version` in the final `SourcingResult`:
```python
return SourcingResult(
    # ... existing fields ...
    prompt_version=graph_result.get("prompt_version", ""),
)
```

**Verification:**
```python
from app.agents.sourcing_agent.prompts import build_decide_prompt

scored = [{"name": "Supplier A", "price": 2.50, "rating": 4.5, "lead_time_days": 7}]
result = build_decide_prompt("USB-C cable", 500, 3.00, scored, [])

assert hasattr(result, 'hash')
assert hasattr(result, 'version')
assert result.version == "v1"
assert len(result.hash) == 16
assert len(result.messages) == 2  # system + user
```

---

## Phase 3 — Hiring Agent Integration

**Goal:** All three LLM-calling nodes in the hiring agent record prompt hash + version.

**Files to modify:**
```
app/agents/hiring_agent/state.py    # Add 6 new fields to AgentState
app/agents/hiring_agent/graph.py    # Update 3 prompt builders + 3 nodes
```

### Step 3a — State fields (`state.py`)

Add six new fields to `AgentState`:
```python
class AgentState(TypedDict, total=False):
    # ... existing fields ...

    # Screen resume node output
    screen_prompt_hash: str         # NEW
    screen_prompt_version: str      # NEW

    # Interview node output
    interview_prompt_hash: str      # NEW
    interview_prompt_version: str   # NEW

    # Offer decision node output
    offer_prompt_hash: str          # NEW
    offer_prompt_version: str       # NEW
```

### Step 3b — Version constants (`graph.py`)

Add at the top of `graph.py`, near the other constants:
```python
# Prompt version tags — bump when you change the corresponding prompt.
SCREEN_PROMPT_VERSION = "v1"
INTERVIEW_PROMPT_VERSION = "v1"
OFFER_PROMPT_VERSION = "v1"
```

### Step 3c — Prompt builders (`graph.py`)

Update all three prompt builders to return `PromptVersion`:

**`_build_screen_prompt`:**
```python
def _build_screen_prompt(role: str, resume_text: str, feedback: str):
    from app.agentmesh.prompt_versioning import version_prompt
    # ... existing prompt construction ...
    return version_prompt(
        [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
        version=SCREEN_PROMPT_VERSION,
    )
```

**`_build_interview_prompt`:**
```python
def _build_interview_prompt(brief, matched_skills, followup_count, transcript):
    from app.agentmesh.prompt_versioning import version_prompt
    # ... existing prompt construction ...
    return version_prompt(
        [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
        version=INTERVIEW_PROMPT_VERSION,
    )
```

**`_build_offer_prompt`:**
```python
def _build_offer_prompt(brief, interview_score: float):
    from app.agentmesh.prompt_versioning import version_prompt
    # ... existing prompt construction ...
    return version_prompt(
        [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
        version=OFFER_PROMPT_VERSION,
    )
```

### Step 3d — Graph nodes (`graph.py`)

Update all three nodes to unpack `PromptVersion`, pass `.messages` to the LLM client, and store `.hash` + `.version` in the state:

**`screen_resume_node`:**
```python
prompt = _build_screen_prompt(brief.role, brief.resume_text, feedback)
response = await client.complete(prompt.messages, temperature=0.0, max_tokens=400)
# ... existing parsing ...
return {
    "skills_matched": matched,
    "screening_notes": summary,
    "screen_prompt_hash": prompt.hash,
    "screen_prompt_version": prompt.version,
}
```

**`interview_node`:**
```python
prompt = _build_interview_prompt(brief, matched, followup_count, transcript)
response = await client.complete(prompt.messages, temperature=0.3, max_tokens=300)
# ... existing parsing ...
return {
    "interview_transcript": new_transcript,
    "followup_count": new_followup_count,
    "interview_score": round(new_score, 1),
    "interview_done": done,
    "interview_prompt_hash": prompt.hash,
    "interview_prompt_version": prompt.version,
}
```

**`offer_decision_node`:**
```python
prompt = _build_offer_prompt(brief, interview_score)
response = await client.complete(prompt.messages, temperature=0.0, max_tokens=300)
# ... existing parsing ...
return {
    "offer_amount": amount,
    "decision_reason": rationale,
    "final_status": "awaiting_offer",
    "offer_prompt_hash": prompt.hash,
    "offer_prompt_version": prompt.version,
}
```

**Verification:**
```python
from app.agents.hiring_agent.graph import _build_screen_prompt, SCREEN_PROMPT_VERSION
result = _build_screen_prompt("Backend Engineer", "Python, FastAPI", "")
assert result.version == SCREEN_PROMPT_VERSION
assert len(result.hash) == 16
```

---

## Phase 4 — Tests

**Goal:** 17 tests covering hash determinism, change detection, state fields, and immutability.

**File to create:**
```
tests/agentmesh/test_prompt_versioning.py
```

**Test categories:**

| Category | Tests | What they verify |
|---|---|---|
| `hash_prompt` | 6 | Determinism, content change, role change, list order, LLMMessage vs dict, hash length |
| `version_prompt` | 2 | Returns PromptVersion with correct fields, immutability (frozen) |
| Sourcing agent | 3 | Returns PromptVersion, hash changes with input, hash is deterministic |
| Hiring agent | 3 | All 3 prompt builders return PromptVersion with correct version tags |
| State fields | 3 | Sourcing AgentState has prompt fields, hiring AgentState has prompt fields, SourcingResult has prompt_version |

**Verification:**
```bash
./venv/bin/python -m pytest tests/agentmesh/test_prompt_versioning.py -v
# Expected: 17 passed
```

---

## Phase 5 — Verification + Commit

**Goal:** Zero regressions, clean commit.

**Steps:**

1. Run the new test suite:
```bash
./venv/bin/python -m pytest tests/agentmesh/test_prompt_versioning.py -v
# Expected: 17 passed
```

2. Run the existing test suite (excluding integration tests that need Temporal/Postgres):
```bash
./venv/bin/python -m pytest \
  tests/agentmesh/test_determinism.py \
  tests/agentmesh/test_verification.py \
  tests/agentmesh/test_tool_registry.py \
  tests/agentmesh/test_circuit_breaker.py \
  tests/agentmesh/test_cost_tracker.py \
  tests/agentmesh/test_egress.py \
  tests/agentmesh/test_fault_injection.py \
  tests/agentmesh/test_output_cap.py \
  tests/agentmesh/test_score_determinism.py \
  tests/agentmesh/test_timeout_boundary.py \
  tests/agents/sourcing_agent/test_graph.py \
  tests/agents/sourcing_agent/test_tool_selection_eval.py \
  -v
# Expected: 73 passed
```

3. Verify the determinism test still passes (prompt versioning doesn't break workflow determinism):
```bash
./venv/bin/python -m pytest tests/agentmesh/test_determinism.py -v
# Expected: 1 passed — no forbidden patterns in workflow.py
```

4. Commit:
```bash
git add app/agentmesh/prompt_versioning.py \
  app/agents/sourcing_agent/state.py \
  app/agents/sourcing_agent/prompts.py \
  app/agents/sourcing_agent/graph.py \
  app/agents/sourcing_agent/workflow.py \
  app/agents/hiring_agent/state.py \
  app/agents/hiring_agent/graph.py \
  tests/agentmesh/test_prompt_versioning.py

git commit -m "Prompt versioning — hash and tag every LLM prompt"
```

**Expected result:** 90/90 tests pass (17 new + 73 existing). Zero regressions. Committed.
