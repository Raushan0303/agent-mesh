# Prompt Versioning — Hash and Tag Every LLM Prompt

**Layer:** Platform (agent-agnostic) + Agent (both sourcing and hiring)
**Duration:** 3 days · ~10 hrs/day · 30 hrs this module
**Status:** COMPLETE — committed in `6bf41a0`

> This document describes the prompt versioning system built on top of the existing AgentMesh platform. Every LLM call now records a `prompt_hash` (SHA256[:16] of the rendered prompt) and `prompt_version` (human-readable tag like `"v1"`) in the LangGraph state. These are checkpointed in Postgres, so every past run has a permanent, queryable record of which prompt was used — without git archaeology.

---

## What to Study (before or alongside building)

- **Deterministic replay**: Why the workflow must be deterministic, and how the activity boundary isolates non-determinism. Reference: `docs/Temporal/workflow.md` Section 6, `blog/02-the-replay-contract.md`.
- **LangGraph checkpointer**: How `AsyncPostgresSaver` persists graph state between node executions, keyed by `thread_id = workflow_id`. Reference: `docs/langgraph/langgraph.md` Section 7.
- **The reproducibility problem**: Why "could not reproduce" is the default for agent systems, and why a log tells you what happened but a record lets you do it again. Reference: Tian Pan, "The Incident Ticket With No Repro Steps."
- **The Internals Question for this module:** Why must the prompt hash be derived from the rendered messages (template + variables), not from the template alone? What does the hash capture that the version tag doesn't?
- **Second Internals Question:** If you change `build_decide_prompt()` and bump `DECIDE_PROMPT_VERSION` from `"v1"` to `"v2"`, what happens to workflows that were started with `"v1"` but are replayed after the code change? (Answer: nothing — the activity result is injected from history, the prompt doesn't re-render. The checkpoint still has `"v1"`. New workflows get `"v2"`.)

---

## What to Build This Module

Everything below is split into **Platform (`agentmesh/`)** vs **Agent (`agents/`)** — the same distinction from `00-PRD.md` Section 3.

### 1. Prompt Versioning Module (Platform — `agentmesh/prompt_versioning.py`)

- A `PromptVersion` dataclass (frozen, immutable) containing `messages`, `hash`, and `version`.
- `hash_prompt()` — deterministic SHA256[:16] of the serialized messages. Accepts both `LLMMessage` dataclass instances and plain dicts. Sorts JSON keys for ordering independence.
- `version_prompt()` — tags a list of messages with its hash and a human-readable version string.

The module is agent-agnostic. It contains zero references to "sourcing", "hiring", "decide", "screen", or any agent-specific concept. It's a utility — any agent's prompt builder calls it.

### 2. Sourcing Agent Integration (Agent — `agents/sourcing_agent/`)

- `prompts.py` — `build_decide_prompt()` returns `PromptVersion` instead of a bare `list[LLMMessage]`. The version constant `DECIDE_PROMPT_VERSION = "v1"` is defined at the top of the file. Bump it when you change the prompt.
- `graph.py` — `decide_node` unpacks the `PromptVersion`, passes `.messages` to `client.complete()`, and stores `.hash` + `.version` in the state dict returned to LangGraph.
- `state.py` — `AgentState` gets two new fields: `prompt_hash: str` and `prompt_version: str`. `SourcingResult` gets `prompt_version: str` so the API response includes which prompt was used.
- `workflow.py` — the final `SourcingResult` includes `prompt_version=graph_result.get("prompt_version", "")`.

### 3. Hiring Agent Integration (Agent — `agents/hiring_agent/`)

- `graph.py` — all three prompt builders (`_build_screen_prompt`, `_build_interview_prompt`, `_build_offer_prompt`) return `PromptVersion`. Three version constants: `SCREEN_PROMPT_VERSION`, `INTERVIEW_PROMPT_VERSION`, `OFFER_PROMPT_VERSION`.
- `state.py` — `AgentState` gets six new fields: `screen_prompt_hash`, `screen_prompt_version`, `interview_prompt_hash`, `interview_prompt_version`, `offer_prompt_hash`, `offer_prompt_version`.

### 4. Tests (Platform — `tests/agentmesh/test_prompt_versioning.py`)

17 tests covering:
- Hash determinism (same messages → same hash, every time)
- Hash change detection (changing content, role, or message order changes the hash)
- `LLMMessage` dataclass vs dict compatibility (both produce the same hash)
- `PromptVersion` immutability (frozen dataclass — can't mutate)
- Sourcing agent: prompt returns `PromptVersion`, hash changes with input, hash is deterministic
- Hiring agent: all three prompt builders return `PromptVersion` with correct version tags
- State fields: both agents' `AgentState` TypedDicts include the new fields
- `SourcingResult` includes `prompt_version` in its output schema

---

## Deliverable

A system where:

1. Every LLM call in both agents records `prompt_hash` and `prompt_version` in the LangGraph state.
2. The LangGraph checkpointer persists these to Postgres, keyed by `thread_id = workflow_id`.
3. Any past run can answer "which prompt was used?" by querying the checkpoint — no git archaeology needed.
4. The `SourcingResult` API response includes `prompt_version` so the caller knows which prompt produced the decision.
5. Changing a prompt is a one-line version bump (`DECIDE_PROMPT_VERSION = "v2"`). Old runs keep `"v1"` in their checkpoint. New runs get `"v2"`.

---

## Outcome (the number you show the interviewer)

- **17/17 new tests pass.** 73/73 existing tests pass. Zero regressions.
- **Every LLM call** in both agents now logs `prompt_hash` and `prompt_version` in structured log lines.
- **The API response** for a completed sourcing workflow includes `"prompt_version": "v1"`.
- **The checkpoint** in Postgres contains the hash + version for every decision, queryable by `thread_id`.

---

## Tests / Evals You Write

1. **Hash determinism test.** Same messages → same hash, every time. Verifies the hash is a pure function of the message content.
2. **Hash change detection test.** Changing a single character in the prompt content changes the hash. Changing the role changes the hash. Reordering messages changes the hash.
3. **LLMMessage vs dict compatibility test.** The same prompt as `LLMMessage` dataclass instances and as plain dicts produces the same hash. This ensures the hash is stable regardless of how the messages are constructed.
4. **PromptVersion immutability test.** The dataclass is frozen — attempting to mutate `.version` raises `AttributeError`.
5. **Sourcing agent prompt test.** `build_decide_prompt()` returns a `PromptVersion` with the correct version tag. Different supplier data produces different hashes. Same inputs produce the same hash.
6. **Hiring agent prompt tests.** All three prompt builders return `PromptVersion` with their respective version tags.
7. **State field tests.** Both agents' `AgentState` TypedDicts include the new prompt hash/version fields. `SourcingResult` includes `prompt_version`.

---

## Interview Talking Point

> "Every LLM call in AgentMesh records a SHA256 hash of the rendered prompt and a human-readable version tag in the LangGraph state. This gets checkpointed in Postgres, so every past run has a permanent record of which prompt was used. When I change a prompt, I bump the version constant — old runs keep their version in the checkpoint, new runs get the new one. This closes the 'could not reproduce' gap: instead of git archaeology to figure out which prompt a past run used, I query the checkpoint. The hash is derived from the rendered messages, not the template — so it captures exactly what the LLM saw, including the variables, not just which template was used. The module is agent-agnostic — any new agent's prompt builder calls `version_prompt()` and gets the same treatment for free."

---

## Concepts Covered This Module

Prompt versioning · content-addressed hashing · SHA256 · deterministic serialization (JSON + sorted keys) · LangGraph state checkpointing · TypedDict field extension · frozen dataclasses · agent-agnostic utility modules · the reproducibility gap in agent systems

---

## Resources

- Tian Pan — "The Incident Ticket With No Repro Steps" (the 02:14 AM support agent, "could not reproduce" as the default)
- Tian Pan — "Exactly-Once Was Hard Before Your Agent Could Retry Itself" (why hashing the request body doesn't work for agents)
- `docs/langgraph/langgraph.md` Section 7 — Checkpointer setup and Postgres persistence
- `blog/02-the-replay-contract.md` — The replay contract and why the LLM must live in an activity
- `app/agentmesh/prompt_versioning.py` — The module itself (87 lines)

---

## Suggested Daily Schedule

| Day | Focus | Output |
|---|---|---|
| 1 | Build the prompt versioning module (`prompt_versioning.py`). Add state fields to both agents. Update sourcing agent prompts + graph + workflow. | Module + sourcing agent integration complete. |
| 2 | Update hiring agent (all 3 prompt builders + 3 nodes + state). Write tests (17 tests covering hash determinism, change detection, state fields, immutability). | Hiring agent integration + test suite complete. |
| 3 | Run existing test suite (73 tests) to verify no regressions. Run new tests (17 tests). Commit. Document. | 90/90 tests pass. Committed. Documented. |
