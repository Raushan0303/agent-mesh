# AgentMesh

**A durable agent execution platform that combines Temporal (crash-safe workflows) with LangGraph (agent reasoning graphs) and a custom MCP tool-calling framework — built to prove that production AI agents need real infrastructure, not just prompt engineering.**

> Status: Weeks 1-6 complete. The platform runs end-to-end with a 5-node LangGraph sourcing agent, pgvector-backed RAG memory, hybrid retrieval (BM25 + dense + RRF + reranker), semantic cache, OpenTelemetry tracing to Jaeger, CI-gated eval harness (50-scenario golden set, LLM-as-judge), feedback loop with drift detection, circuit breakers, bulkheads, per-Activity timeouts, and 39 passing tests. The whole system is live, documented, and CI-gated.

---

## Why this project exists

Most AI agent demos are a Python script with a prompt and an API key. They work until:
- The process crashes and the agent loses all state
- A tool call returns malformed JSON and the agent silently breaks
- A side-effecting tool (payment, order) gets retried and creates duplicates
- You need to change the agent's logic while workflows are mid-flight

AgentMesh solves these with **real infrastructure**: Temporal for durable execution, LangGraph for explicit graph topology, and a custom tool registry with schema validation and sandboxing — the layer that doesn't exist in either Temporal or LangGraph and has to be built from scratch.

The first agent is a **sourcing agent** that queries a mock marketplace for suppliers, gets price quotes, checks ratings, creates purchase orders, and initiates payments — all with human-in-the-loop checkpoints and exactly-once side effects.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Folder Structure](#folder-structure)
- [Platform vs Agent Split](#platform-vs-agent-split)
- [Key Engineering Decisions](#key-engineering-decisions)
- [The Tool Registry — the novel engineering](#the-tool-registry--the-novel-engineering)
- [Idempotency — why Temporal isn't enough](#idempotency--why-temporal-isnt-enough)
- [Workflow Versioning — changing code without breaking in-flight workflows](#workflow-versioning--changing-code-without-breaking-in-flight-workflows)
- [Progress: What's Built So Far](#progress-whats-built-so-far)
- [Test Results](#test-results)
- [API Reference](#api-reference)
- [Glossary](#glossary)

---

## Quick Start

### Prerequisites

- Docker (for Temporal + Postgres)
- Python 3.11+
- `pip install -r requirements.txt`

### Start the stack (3 terminals)

```bash
# Terminal 1: infrastructure
docker-compose up -d

# Initialize the agentmesh database (one-time)
source venv/bin/activate
PYTHONPATH=. python scripts/init_agent_db.py

# Terminal 2: worker
source venv/bin/activate
python -m app.agentmesh.worker_runner

# Terminal 3: gateway
source venv/bin/activate
python server.py
```

### Submit a sourcing request

```bash
# Health check
curl http://localhost:8000/health

# Start a sourcing workflow
curl -X POST http://localhost:8000/workflows \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "sourcing_agent",
    "input": {
      "item": "USB-C cable",
      "quantity": 500,
      "budget": 3.00
    }
  }'

# Check status
curl http://localhost:8000/workflows/<workflow_id>
```

### View in Temporal Web UI

Open http://localhost:8080 — search for your workflow ID to see the full event timeline.

### Run tests

```bash
# Full suite (39 tests, ~3min)
python -m pytest tests/ -v --ignore=tests/agentmesh/test_chaos_idempotency.py

# CI-gated eval suite (50 scenarios)
PYTHONPATH=. python -m app.agents.sourcing_agent.eval_glue

# Populate memory store (one-time)
PYTHONPATH=. python scripts/init_memory_store.py

# Locust load test (500 concurrent users)
locust -f tests/agentmesh/locustfile.py --headless \
    --users 500 --spawn-rate 10 --run-time 60s \
    --host http://localhost:8000
```

### View traces in Jaeger

Open http://localhost:16686 — select the `agentmesh` service to see end-to-end traces.

---

## Architecture Overview

```
HTTP Request (POST /workflows)
    │
    ▼
FastAPI Gateway ─────────────────── app/agentmesh/gateway/
    │ looks up agent_type in AGENT_REGISTRY (in-memory dict)
    │ validates input against the agent's Pydantic model
    │ calls client.start_workflow() → Temporal
    │
    ▼
Temporal Server ─────────────────── docker-compose (self-hosted)
    │ Frontend + History + Matching + Persistence (Postgres)
    │ Records every step as an event in the Event History
    │ Redelivers Activity Tasks if a Worker crashes
    │
    ▼
Worker Process ──────────────────── app/agentmesh/worker_runner.py
    │ Polls a task queue, picks up Workflow + Activity tasks
    │
    ├── SourcingWorkflow.run() ──── app/agents/sourcing_agent/workflow.py
    │     │ Calls Activities in sequence with per-Activity retry policies
    │     │
    │     ├── Activity 1: run_research_activity
    │     │     └── LangGraph StateGraph ── app/agents/sourcing_agent/graph.py
    │     │           └── Research node calls tools through the Tool Registry
    │     │                 ├── query_suppliers
    │     │                 ├── get_price_quote
    │     │                 └── check_seller_rating
    │     │
    │     ├── [Versioned] Activity 2: score_suppliers_activity
    │     │     └── Ranks suppliers by price/rating/lead-time
    │     │
    │     ├── Activity 3: create_po_activity
    │     │     └── Tool Registry → create_purchase_order (idempotent, Postgres-backed)
    │     │
    │     └── Activity 4: initiate_payment_activity
    │           └── Tool Registry → initiate_payment (idempotent, Postgres-backed)
    │
    └── Tool Registry ───────────── app/agentmesh/tool_registry/
          │ Validates input against Pydantic schema before execution
          │ Enforces per-tool timeout (kills hung tools)
          │ Validates output against Pydantic schema after execution
          │ Wraps all errors in structured exceptions
          └── ZERO knowledge of what any tool actually does
```

### The five layers

| Layer | What it is | Owner | Built in |
|---|---|---|---|
| **Gateway** | FastAPI HTTP API, agent-agnostic routing | Platform | Week 1 |
| **Temporal** | Durable execution, crash recovery, event history | Dependency (self-hosted) | Week 1 |
| **LangGraph** | Agent state graph, nodes, edges, shared state | Dependency | Week 1 |
| **Tool Registry** | Schema validation, timeout, sandboxing for tool calls | **Platform (novel)** | Week 2 |
| **Eval Harness** | Tool-selection accuracy scoring | Platform | Week 2 |

---

## Folder Structure

```
agent-mesh/
├── server.py                              # FastAPI entry point (uvicorn)
├── docker-compose.yml                     # Temporal + Postgres + Web UI
├── requirements.txt                       # Python dependencies
├── pyproject.toml                         # Project config (pytest, ruff)
│
├── app/
│   ├── main.py                            # FastAPI app + router + agent import (triggers self-registration)
│   │
│   ├── core/                              # ── Shared config ──
│   │   ├── config.py                      #   Settings via pydantic-settings (env vars, AGENTMESH_ prefix)
│   │   └── constants.py                   #   Retry policy templates + timeout defaults
│   │
│   ├── agentmesh/                         # ── PLATFORM (agent-agnostic) ──
│   │   │                                    # grep -r "sourcing" app/agentmesh/ → ZERO results
│   │   │
│   │   ├── gateway/                       #   HTTP gateway
│   │   │   ├── registry.py                #     AgentRegistration dataclass + AGENT_REGISTRY dict
│   │   │   ├── routes.py                  #     POST /workflows, GET /workflows/{id}, POST /workflows/{id}/signal
│   │   │   └── models.py                  #     Generic request/response envelopes
│   │   │
│   │   ├── temporal/                      #   Temporal integration
│   │   │   ├── client.py                  #     Singleton Temporal client factory
│   │   │   ├── worker.py                  #     run_worker() — accepts workflows + activities as params
│   │   │   └── versioning.py              #     is_patched() + deprecate_patch() helpers
│   │   │
│   │   ├── tool_registry/                 #   MCP tool-calling framework (the novel layer)
│   │   │   ├── spec.py                    #     ToolSpec dataclass (name, schemas, timeout, idempotency)
│   │   │   ├── exceptions.py              #     5 custom exceptions
│   │   │   ├── schema_validation.py       #     Pydantic v2 input/output validation
│   │   │   ├── sandbox.py                 #     asyncio timeout enforcement
│   │   │   ├── registry.py                #     ToolRegistry class — register() + call()
│   │   │   └── __init__.py                #     Public API exports
│   │   │
│   │   ├── evals/                         #   Eval harness (agent-agnostic)
│   │   │   ├── models.py                  #     EvalScenario + EvalResult dataclasses
│   │   │   └── harness.py                 #     EvalHarness — run(dataset, invoke_agent)
│   │   │
│   │   └── worker_runner.py               #   Entry point — the ONLY platform file that imports agent code
│   │
│   └── agents/                            # ── AGENT IMPLEMENTATIONS ──
│       └── sourcing_agent/                #   The first (and so far only) agent
│           ├── __init__.py                #     TASK_QUEUE + self-registration into AGENT_REGISTRY + tool registration
│           ├── state.py                   #     Pydantic input/output models + LangGraph state TypedDict
│           ├── mock_marketplace.py        #     8 mock suppliers across 4 item types
│           ├── tool_schemas.py            #     Pydantic input/output models for all 5 tools
│           ├── tools.py                   #     5 tool implementations + register_sourcing_tools()
│           ├── graph.py                   #     LangGraph StateGraph (Research node + retry loop)
│           ├── activity.py                #     4 Temporal Activities (research, score, create_po, payment)
│           ├── workflow.py                #     SourcingWorkflow — calls Activities with per-Activity retry policies
│           ├── db.py                      #     asyncpg pool + idempotency table helpers
│           └── eval_dataset.py            #     20 sourcing scenarios for the eval harness
│
├── tests/
│   ├── agentmesh/                         #   Platform tests
│   │   ├── test_crash_recovery.py         #     Kill Worker mid-Activity → workflow completes on new Worker
│   │   ├── test_determinism.py            #     Static scan: no non-deterministic code in Workflow
│   │   ├── test_tool_registry.py          #     9 tests: timeout, schema validation, error wrapping
│   │   ├── test_chaos_idempotency.py      #     20-trial chaos test: 0 duplicate POs, 0 duplicate payments
│   │   └── test_workflow_versioning.py    #     Versioning: new workflows use scored path (4 Activities)
│   └── agents/sourcing_agent/
│       ├── test_graph.py                  #     Stop condition tests (success + no-match paths)
│       └── test_tool_selection_eval.py    #     20-scenario eval: 100% tool-selection accuracy
│
└── scripts/
    ├── hello_workflow.py                  #   Throwaway smoke test (Phase 0)
    ├── init_agent_db.py                   #   One-time DB + table creation
    └── e2e_test.py                        #   E2E performance test + event history capture
```

---

## Platform vs Agent Split

The most important architectural decision in this project: **the platform knows nothing about the agent.**

```
app/agentmesh/     ← PLATFORM (agent-agnostic)
  gateway/           → routes by agent_type string, never imports agent code
  temporal/          → worker accepts workflows + activities as parameters
  tool_registry/     → validates schemas and enforces timeouts, doesn't know what tools do
  evals/             → scores tool-selection accuracy, doesn't know which tools are correct

app/agents/        ← AGENT (domain-specific)
  sourcing_agent/    → registers itself, provides tools, defines workflow logic
```

**Verified by grep:**
```bash
grep -r "sourcing\|supplier\|purchase" app/agentmesh/gateway/         → ZERO results ✅
grep -r "sourcing\|supplier\|purchase" app/agentmesh/tool_registry/    → ZERO results ✅
grep -r "sourcing\|supplier\|purchase" app/agentmesh/evals/            → ZERO results ✅
grep -r "sourcing\|supplier\|purchase" app/agentmesh/temporal/         → ZERO results ✅
grep -r "sourcing\|supplier\|purchase" app/core/constants.py           → ZERO results ✅
```

Adding a second agent touches **zero files** in `app/agentmesh/`. You create `app/agents/new_agent/`, add one import line in `main.py`, and the gateway, worker, tool registry, and eval harness all work without modification.

---

## Key Engineering Decisions

### 1. Why Temporal + LangGraph, not just one?

| Concern | Temporal | LangGraph |
|---|---|---|
| Durable execution (crash recovery) | ✅ Event History, replay | ❌ |
| Agent state graph (nodes, edges) | ❌ | ✅ StateGraph |
| Non-deterministic work (LLM calls, HTTP) | ✅ Activities isolate it | ❌ |
| Human-in-the-loop (pause + resume) | ✅ Signals + wait_condition | ❌ |
| Retry policies (per-activity) | ✅ Configurable | ❌ |

Running a LangGraph graph inside a Temporal Activity gets you both: Temporal owns durability, LangGraph owns topology. The LLM call lives inside the Activity (non-deterministic code is allowed there), not in the Workflow function (which must be deterministic for replay to work).

### 2. Why must Workflow code be deterministic?

Temporal reconstructs a Workflow's state by **replaying** its Event History. If the Workflow code does something non-deterministic (calls `datetime.now()`, uses `random()`, makes an HTTP request), the replay produces different results than the original execution — and Temporal can't reconcile the two. This causes a "non-deterministic error" and the Workflow gets stuck.

**The rule:** all non-deterministic work (LLM calls, tool calls, HTTP, DB access) goes inside Activities. The Workflow function only orchestrates: "call Activity A, then Activity B, then Activity C." This is enforced by a [static scan test](tests/agentmesh/test_determinism.py) that greps the Workflow file for forbidden patterns.

### 3. Why a plugin pattern for agent registration?

The gateway stays agent-agnostic. Adding a new agent = create `app/agents/new_agent/` + add one import line in `main.py`. The gateway code never changes. The agent self-registers at import time:

```python
# app/agents/sourcing_agent/__init__.py
AGENT_REGISTRY["sourcing_agent"] = AgentRegistration(
    workflow_class=SourcingWorkflow,
    task_queue=TASK_QUEUE,
    input_model=SourcingBriefInput,
)
```

The gateway looks up `agent_type` in `AGENT_REGISTRY` and routes accordingly. It has no idea what "sourcing_agent" means — it just sees a string key.

### 4. Why split the Workflow into multiple Activities?

A single Activity wrapping the entire graph works (Week 1 did this), but it means there's no Temporal-level checkpoint between operations. The chaos test needs to kill the Worker **between** PO creation and payment — that requires a Temporal boundary between them.

Splitting into separate Activities gives Temporal a checkpoint: the PO Activity completes and reports to Temporal before the payment Activity starts. If the Worker dies in between, Temporal knows the PO is done and only retries the payment.

### 5. Why two retry policy templates?

| Template | Max Attempts | Non-retryable | Used for |
|---|---|---|---|
| `AGGRESSIVE_RETRY_TEMPLATE` | 10 | (none) | Read-only tools (queries, ratings) |
| `STRICT_NON_RETRYABLE_TEMPLATE` | 3 | `ToolExecutionError` | Side-effecting tools (orders, payments) |

Read-only tools are safe to retry aggressively — if `query_suppliers` fails and retries, nothing bad happens. Side-effecting tools are different: if `initiate_payment` fails ambiguously (timeout after the payment was sent), a blind retry could create a duplicate. The strict template retries fewer times and doesn't retry on `ToolExecutionError` (the tool itself raised — an ambiguous failure that might mean the side effect already happened).

### 6. Why asyncio timeout instead of subprocess isolation (for now)?

Week 2's sandbox is timeout-only. Full subprocess isolation (resource limits, filesystem blocking) is a future enhancement. The interface (`run_with_timeout`) stays the same; the implementation gets stronger. This is intentional sequencing — prove the timeout works first, then add the harder isolation.

---

## The Tool Registry — the novel engineering

This is the **single harness layer with no off-the-shelf equivalent**. Temporal doesn't validate tool inputs. LangGraph doesn't sandbox tool calls. If a model emits malformed JSON arguments to `initiate_payment`, neither framework stops it.

### What the Tool Registry does

Every tool call goes through this flow:

```
LangGraph node calls registry.call("create_purchase_order", args)
  │
  ├── 1. Look up tool by name → ToolNotFoundError if missing
  ├── 2. Validate input against Pydantic model → SchemaValidationError if bad
  ├── 3. Execute with timeout → ToolTimeoutError if hung
  ├── 4. Validate output against Pydantic model → SchemaValidationError if bad
  └── 5. Return validated result
```

### Why this matters

A bare prompt instruction ("always return valid JSON") doesn't enforce anything — the model can ignore it. Schema validation is a **hard gate**: if the args don't match the schema, the call never happens. This is the layer that prevents a corrupted model output from calling `initiate_payment` with a bad amount.

### Adding a new tool

A config entry, not a code change inside `agentmesh/tool_registry/`:

```python
registry.register(
    ToolSpec(
        name="create_purchase_order",
        input_model=CreatePurchaseOrderInput,   # Pydantic model
        output_model=CreatePurchaseOrderOutput,  # Pydantic model
        timeout_seconds=15.0,
        idempotency_required=True,
    ),
    create_purchase_order_impl,                  # async function
)
```

The registry has zero knowledge of what `create_purchase_order` does. It only knows there's a name, a schema, and a timeout.

---

## Idempotency — why Temporal isn't enough

Temporal guarantees crash-safe replay: if a Worker dies, the Activity Task is redelivered to another Worker. But **Temporal doesn't know what the Activity did** — it only knows the Activity didn't report completion. If the Activity created a purchase order before crashing, Temporal will retry it, and without idempotency, you get **two purchase orders**.

### The solution: check-before-execute

```python
async def create_purchase_order_impl(supplier_name, item, quantity, unit_price):
    workflow_id = activity.info().workflow_id
    idempotency_key = f"{workflow_id}:create_po"

    # Check if we already created this PO (from a previous retry)
    existing = await conn.fetchrow(
        "SELECT po_id, status FROM sourcing_agent_purchase_orders WHERE idempotency_key = $1",
        idempotency_key,
    )
    if existing:
        return {"po_id": existing["po_id"], "status": existing["status"]}  # Return existing, don't duplicate

    # Create the PO
    po_id = f"PO-{uuid.uuid4().hex[:8]}"
    await conn.execute("INSERT INTO sourcing_agent_purchase_orders ...", ...)
    return {"po_id": po_id, "status": "created"}
```

The idempotency key is derived from `workflow_id + step_name`. On retry, the same key is derived, the existing row is found, and the duplicate is prevented.

### The proof: 20-trial chaos test

```
============================================================
Chaos Test Results: 20 trials
  Total POs:      20 (expected 20)
  Total Payments: 20 (expected 20)
  Duplicate POs:      0
  Duplicate Payments: 0
============================================================
```

Each trial: submit a workflow → wait for PO to appear in DB → SIGKILL the Worker → start a new Worker → wait for completion → assert exactly 1 PO and 1 payment. **20 trials, 0 duplicates.**

### Why the platform can't do this for you

The platform can't make `create_purchase_order` idempotent because it doesn't know what "create_purchase_order" means. The idempotency key and the check-before-execute pattern require knowing which table to check and what constitutes a "duplicate." The platform provides retry policy *templates*; the agent provides the *guarantee*.

---

## Workflow Versioning — changing code without breaking in-flight workflows

The problem: you have a Workflow running in production. You need to add a new step (e.g., a scoring Activity). If you just change the code, in-flight Workflows will replay against the new code and fail (non-deterministic error — the replay doesn't match the original execution).

### The solution: `workflow.patched()`

```python
# In the Workflow:
if is_patched("add-score-step"):
    # NEW path: run score Activity (newly started workflows)
    scored = await workflow.execute_activity(score_suppliers_activity, ...)
    best_supplier = scored.suppliers[0]
else:
    # OLD path: pick cheapest directly (in-flight workflows from before the change)
    best_supplier = min(suppliers, key=lambda s: s["price"])
```

- **New workflows:** `is_patched()` returns True → score Activity runs → 4 total Activities
- **Old in-flight workflows:** `is_patched()` returns False → skip scoring → 3 total Activities
- Temporal records the patch decision as a marker in the event history. On replay, the marker is checked.

### The proof

The versioning test shows 4 Activities completed (research → score → create_po → initiate_payment), proving the new code path is active for new workflows.

---

## Progress: What's Built So Far

### Week 1 — Temporal + LangGraph Foundations

- Docker Compose stack (Temporal + Postgres + Web UI)
- AgentMesh Gateway (FastAPI, agent-agnostic routing via `AGENT_REGISTRY`)
- AgentMesh Worker Process (accepts workflows + activities as config)
- Sourcing Agent: minimal LangGraph graph with one Research node
- Stop condition: succeed after 1 supplier found, halt after 3 empty attempts
- Crash recovery test: kill Worker mid-Activity → workflow completes on new Worker
- Determinism scan: no forbidden patterns in Workflow code
- **E2E latency: p50 = 0.32s**

### Week 2 — MCP Tool-Calling Framework

- **Tool Registry** (`agentmesh/tool_registry/`) — the novel layer:
  - `ToolSpec` dataclass (name, Pydantic input/output models, timeout, idempotency flag)
  - Schema validation (Pydantic v2, input + output)
  - Sandbox (asyncio timeout enforcement)
  - 5 custom exceptions (ToolNotFoundError, SchemaValidationError, ToolTimeoutError, ToolExecutionError, ToolRegistryError)
- 5 sourcing tools registered: `query_suppliers`, `get_price_quote`, `check_seller_rating`, `create_purchase_order`, `initiate_payment`
- Multi-tool sequence in Research node: query → quote → rating (all through the registry)
- **Eval Harness** (`agentmesh/evals/`) — agent-agnostic, scores tool-selection accuracy
- 20-scenario eval dataset: **100% tool-selection accuracy**
- 9 fault-injection tests: timeout, schema validation, error wrapping, duplicate registration
- **E2E latency: p50 = 0.32s** (unchanged — the registry adds <1ms overhead)

### Week 3 — Retries, Idempotency & Workflow Versioning

- Two retry policy templates: aggressive (10 attempts, for reads) + strict non-retryable (3 attempts, for side effects)
- Workflow Versioning helper: `is_patched()` + `deprecate_patch()` wrappers
- Postgres-backed idempotency: `sourcing_agent_purchase_orders` + `sourcing_agent_payment_intents` tables
- Idempotent tool implementations: check-before-execute with keys derived from `workflow_id + step_name`
- Split Workflow into 4 Activities with per-Activity retry policies
- Versioned score step: new workflows get scoring, old in-flight workflows don't
- **Chaos test: 20 trials, 0 duplicate POs, 0 duplicate payments**
- **Versioning test: 4 Activities (scored path active)**

### Week 4 — Full 5-Node Graph + Human-in-the-Loop + Tenant Isolation

- Full 5-node LangGraph graph: Research → Score → Decide → Approve → Confirm
- Human-in-the-loop checkpoint via LangGraph's `interrupt()` at the Approve node
- Gateway `/approve` endpoint sends approval signal to resume the graph
- Tenant isolation via Temporal namespaces (per-tenant task queues)
- Graph kill at Approve node → resume at Approve node (checkpoint recovery)
- Score determinism test: same inputs → same scored suppliers

### Week 5 — Reliability Engineering (Circuit Breakers, Bulkheads, Timeouts)

- Per-tool circuit breaker (CLOSED → OPEN → HALF_OPEN), keyed by tool name
- Bulkhead: concurrent call limit per downstream service
- Per-Activity timeouts: StartToClose, ScheduleToStart, ScheduleToClose
- Fault injection harness: configurable per-tool failure rate
- DLQ CLI: inspect and retry failed workflows via Temporal Visibility API
- **Circuit breaker fail-fasts 54% of calls under 40% failure rate**
- **Activity killed at 2s (StartToClose), not 8s (ScheduleToClose)**

### Week 6 — Production RAG, Evals & Observability (Capstone)

- **pgvector memory store** — cross-session memory, separate from Temporal Event History and LangGraph checkpoints
- **Hybrid retrieval** — BM25 (Postgres FTS) + dense (pgvector) fused via Reciprocal Rank Fusion, then cross-encoder reranker
- **Semantic cache** — cache LLM responses keyed by embedding similarity, with TTL/staleness policy
- **OpenTelemetry tracing** — stitches Temporal + LangGraph + tool calls into one trace per request, exported to Jaeger
- **CI-gated eval harness** — 50-scenario golden dataset, tool accuracy, completion rate, RAG recall@k, LLM-as-judge
- **Feedback loop** — thumbs up/down capture, weekly drift report comparing eval scores and acceptance rate
- **Locust load test** — 500 concurrent sourcing briefs, p50/p95/p99 latency
- **Full regression pack** — re-runs all Week 3-6 chaos tests as one nightly suite
- **CI gate: 100% tool accuracy, 100% completion rate, 100% judge score — PASSED**

### What's next

All 6 weeks complete. The platform is demo-able and CI-gated.

---

## Test Results

### Full suite (39 tests)

| Test | What It Proves | Result |
|---|---|---|
| `test_ci_eval_suite_passes_gate` | 50/50 scenarios pass, CI gate PASSED | PASSED |
| `test_eval_dataset_has_50_scenarios` | Golden dataset has 50 scenarios | PASSED |
| `test_semantic_cache_hit_and_miss` | Cache hits on identical, misses on dissimilar | PASSED |
| `test_hybrid_retrieval_recall` | BM25 + Dense + RRF + Reranker works | PASSED |
| `test_feedback_store_and_drift_report` | Feedback + drift detection works | PASSED |
| `test_circuit_breaker_*` (9 tests) | CLOSED→OPEN→HALF_OPEN transitions, fail-fast, per-tool independence | PASSED |
| `test_fault_injection_*` (2 tests) | Breaker fail-fasts 54% under 40% failure rate, bulkhead isolation | PASSED |
| `test_timeout_boundary_*` (2 tests) | Activity killed at 2s (StartToClose), not 8s (ScheduleToClose) | PASSED |
| `test_crash_recovery` | Kill Worker mid-Activity → workflow completes on new Worker | PASSED |
| `test_determinism` | No non-deterministic code in Workflow (static scan) | PASSED |
| `test_tool_registry_*` (8 tests) | Schema validation, timeout, sandbox, errors | PASSED |
| `test_workflow_versioning` | New workflow uses scored path (4 Activities) | PASSED |
| `test_graph_kill_at_approve` | Graph killed at Approve → resumes at Approve | PASSED |
| `test_namespace_isolation` | Tenant A invisible to Tenant B | PASSED |
| `test_score_determinism` | Score node is deterministic | PASSED |
| `test_regression_pack_smoke` | All Week 3-6 test modules importable | PASSED |
| `test_research_*` (3 tests) | Graph Research node: finds suppliers, no-match halts, HDMI works | PASSED |
| `test_tool_selection_eval` | 50 scenarios → 100% tool-selection accuracy | PASSED |

### CI-gated eval scorecard

```
============================================================
CI EVAL SCORECARD
============================================================
  Scenarios:          50
  Passed:             50/50
  Pass rate:          100.00%
  Avg tool accuracy:  100.00%
  Completion rate:    100.00%
  Avg judge score:    100.00%
============================================================

  CI GATE PASSED ✅
    Thresholds: tool_accuracy ≥ 90%, completion_rate ≥ 95%, rag_recall ≥ 80%
```

---

## API Reference

Base URL: `http://localhost:8000`

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Gateway liveness check |
| `POST` | `/workflows` | Start a new workflow execution |
| `GET` | `/workflows/{workflow_id}` | Get workflow status |
| `POST` | `/workflows/{workflow_id}/signal` | Send a signal to a running workflow |

### POST /workflows

```json
{
  "agent_type": "sourcing_agent",
  "input": {
    "item": "USB-C cable",
    "quantity": 500,
    "budget": 3.00,
    "deadline": "2026-08-20"
  }
}
```

**Response:** `{"workflow_id": "sourcing_agent-abc12345", "agent_type": "sourcing_agent"}`

### GET /workflows/{workflow_id}

**Response:** `{"workflow_id": "sourcing_agent-abc12345", "status": "COMPLETED", "run_id": "..."}`

---

## Glossary

**Temporal** — An open-source durable execution system. It records every step of a workflow as an event in a database, so if the process crashes, the workflow resumes from where it left off. Think of it as a database that remembers what your code was doing.

**LangGraph** — A library for building agent reasoning as a graph. Nodes are steps (research, scoring, decision), edges are transitions. The state object flows between nodes. Think of it as a state machine for AI agents.

**Activity** — A Temporal concept. The place where all non-deterministic work lives (LLM calls, HTTP requests, DB access). Activities can fail and retry without breaking the Workflow's replay guarantee.

**Workflow** — A Temporal concept. The orchestration code that calls Activities in sequence. Must be deterministic — no `datetime.now()`, no `random()`, no HTTP calls. Temporal replays this code to reconstruct state after a crash.

**Task Queue** — A named mailbox for Temporal tasks. The client submits a workflow to a task queue; Workers poll that queue for tasks. Both must use the same queue name.

**Event History** — The permanent, append-only record of every step in a Workflow's execution. Stored in Postgres. This is what makes Temporal durable — the history is the source of truth, not the process's memory.

**Replay** — Temporal's crash recovery mechanism. When a Worker picks up a Workflow after a crash, it replays the Event History through the Workflow code to reconstruct the state. This is why Workflow code must be deterministic.

**MCP (Model Context Protocol)** — A protocol for connecting AI models to external tools and data sources. AgentMesh uses MCP concepts (tool registration, schema-based invocation) in its Tool Registry.

**Idempotency Key** — A unique string that identifies a specific side-effecting operation. If the same key is seen twice (e.g., from a retry), the operation is not repeated. Derived from `workflow_id + step_name` in AgentMesh.

**Schema Validation** — Checking that data matches a defined structure (a Pydantic model) before using it. Prevents malformed inputs from reaching tool implementations.

**Circuit Breaker** — A reliability pattern that stops calling a failing service after N consecutive failures, giving it time to recover. (Planned for Week 5.)

**Eval Harness** — A testing framework that runs an agent against a fixed set of scenarios and scores the results. AgentMesh's harness measures tool-selection accuracy (did the agent call the right tools in the right order?).

**Workflow Versioning** — A mechanism for changing Workflow code without breaking Workflows that are already running. Uses `workflow.patched()` to branch between old and new code paths based on whether the Workflow was started before or after the change.

**Human-in-the-Loop** — A design pattern where the AI prepares and recommends, but a human gives the final confirmation before anything real happens. (Planned for Week 4 — the `Approve` node with LangGraph's `interrupt()`.)

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Durable execution | Temporal (self-hosted) | Crash-safe replay, event history, per-activity retry policies |
| Agent graph | LangGraph | Explicit node/edge topology, shared state, checkpointing |
| Tool protocol | MCP (Model Context Protocol) | Standardized tool registration and invocation |
| HTTP API | FastAPI + Uvicorn | Async, Pydantic-native, auto-docs |
| Database | PostgreSQL 16 | Temporal's event store + agent idempotency tables |
| Schema validation | Pydantic v2 | Type coercion, custom validators, strict mode |
| DB driver | asyncpg | Fast async PostgreSQL access for Python |
| Testing | pytest + pytest-asyncio | Async test support, fixture management |

---

## License

This project is part of a personal portfolio. See the repository for details.
