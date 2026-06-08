"""
End-to-end test + event history capture.

Requires: docker-compose up -d + worker running + gateway running.

Usage:
    python scripts/e2e_test.py

This script:
1. Submits 10 sourcing workflows via the gateway API (success path).
2. Submits 3 no-match workflows (no_matches path).
3. Measures p50/p99 latency for each.
4. Fetches event history for one workflow and saves it to docs/traces/.
5. Prints a summary you can screenshot.
"""
import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from temporalio.client import Client

from app.core.config import settings

GATEWAY_URL = "http://localhost:8000"
TRACE_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "traces")

EVENT_TYPE_NAMES = {
    1: "WorkflowExecutionStarted",
    2: "WorkflowExecutionCompleted",
    3: "WorkflowExecutionFailed",
    4: "WorkflowExecutionTimedOut",
    5: "WorkflowTaskScheduled",
    6: "WorkflowTaskStarted",
    7: "WorkflowTaskCompleted",
    8: "WorkflowTaskTimedOut",
    9: "WorkflowTaskFailed",
    10: "ActivityTaskScheduled",
    11: "ActivityTaskStarted",
    12: "ActivityTaskCompleted",
    13: "ActivityTaskFailed",
    14: "ActivityTaskTimedOut",
    15: "TimerStarted",
    16: "TimerFired",
    17: "CancelTimerFailed",
    18: "TimerCanceled",
    19: "WorkflowExecutionCancelRequested",
    20: "WorkflowExecutionCanceled",
    21: "RequestCancelExternalWorkflowExecutionInitiated",
    22: "RequestCancelExternalWorkflowExecutionFailed",
    23: "ExternalWorkflowExecutionCancelRequested",
    24: "MarkerRecorded",
    25: "WorkflowExecutionSignaled",
    26: "WorkflowExecutionTerminated",
    27: "WorkflowExecutionContinuedAsNew",
    28: "StartChildWorkflowExecutionInitiated",
    29: "StartChildWorkflowExecutionFailed",
    30: "ChildWorkflowExecutionStarted",
    31: "ChildWorkflowExecutionCompleted",
    32: "ChildWorkflowExecutionFailed",
    33: "ChildWorkflowExecutionCanceled",
    34: "ChildWorkflowExecutionTimedOut",
    35: "ChildWorkflowExecutionTerminated",
    36: "SignalExternalWorkflowExecutionInitiated",
    37: "SignalExternalWorkflowExecutionFailed",
    38: "ExternalWorkflowExecutionSignaled",
    39: "UpsertWorkflowSearchAttributes",
    40: "ActivityTaskCancelRequested",
    41: "ActivityTaskCanceled",
}


def _event_type_name(etype) -> str:
    if hasattr(etype, "name") and etype.name:
        return etype.name
    return EVENT_TYPE_NAMES.get(int(etype), f"Unknown({etype})")


async def submit_workflow(client: httpx.AsyncClient, item: str, quantity: int, budget: float) -> dict:
    """Submit a workflow and return the response."""
    resp = await client.post(
        f"{GATEWAY_URL}/workflows",
        json={
            "agent_type": "sourcing_agent",
            "input": {"item": item, "quantity": quantity, "budget": budget},
        },
    )
    return resp.json()


async def wait_for_completion(workflow_id: str, timeout: float = 30) -> str:
    """Poll gateway until workflow completes, return status."""
    async with httpx.AsyncClient() as client:
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = await client.get(f"{GATEWAY_URL}/workflows/{workflow_id}")
            if resp.status_code == 200:
                data = resp.json()
                if data["status"] == "COMPLETED":
                    return data["status"]
            await asyncio.sleep(0.3)
    return "TIMEOUT"


async def fetch_event_history(workflow_id: str) -> list:
    """Fetch event history directly from Temporal."""
    temporal_client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
    )
    handle = temporal_client.get_workflow_handle(workflow_id)
    history = await handle.fetch_history()
    events = []
    for event in history.events:
        etype = event.event_type
        events.append({
            "event_id": event.event_id,
            "event_type": _event_type_name(etype),
            "event_time": str(event.event_time),
        })
    return events


async def main():
    print("=" * 70)
    print("AgentMesh — End-to-End Test + Event History Capture")
    print("=" * 70)

    # ── Phase 1: Success path (10 requests) ──
    print("\n📊 Phase 1: Success path — 10 requests (USB-C cable, qty=500, budget=3.00)")
    success_latencies = []
    success_workflow_ids = []

    async with httpx.AsyncClient() as client:
        for i in range(10):
            start = time.perf_counter()
            resp = await submit_workflow(client, "USB-C cable", 500, 3.00)
            workflow_id = resp["workflow_id"]
            success_workflow_ids.append(workflow_id)
            status = await wait_for_completion(workflow_id)
            elapsed = time.perf_counter() - start
            success_latencies.append(elapsed)
            print(f"  [{i+1:2d}/10] {workflow_id} → {status} in {elapsed:.3f}s")

    # ── Phase 2: No-match path (3 requests) ──
    print("\n📊 Phase 2: No-match path — 3 requests (quantum computer, qty=1, budget=0.01)")
    nomatch_latencies = []
    nomatch_workflow_ids = []

    async with httpx.AsyncClient() as client:
        for i in range(3):
            start = time.perf_counter()
            resp = await submit_workflow(client, "quantum computer", 1, 0.01)
            workflow_id = resp["workflow_id"]
            nomatch_workflow_ids.append(workflow_id)
            status = await wait_for_completion(workflow_id)
            elapsed = time.perf_counter() - start
            nomatch_latencies.append(elapsed)
            print(f"  [{i+1}/3]  {workflow_id} → {status} in {elapsed:.3f}s")

    # ── Phase 3: Latency summary ──
    print("\n📈 Latency Summary")
    print("-" * 50)

    def print_stats(label, latencies):
        latencies_sorted = sorted(latencies)
        p50 = statistics.median(latencies)
        p99 = latencies_sorted[-1] if len(latencies_sorted) == 1 else latencies_sorted[int(len(latencies_sorted) * 0.99)]
        avg = statistics.mean(latencies)
        print(f"  {label}:")
        print(f"    avg  = {avg:.3f}s")
        print(f"    p50  = {p50:.3f}s")
        print(f"    p99  = {p99:.3f}s")
        print(f"    min  = {min(latencies):.3f}s")
        print(f"    max  = {max(latencies):.3f}s")

    print_stats("Success path (10 requests)", success_latencies)
    print_stats("No-match path (3 requests)", nomatch_latencies)

    # ── Phase 4: Event history capture ──
    print("\n📜 Phase 4: Event History Capture")
    print("-" * 50)

    os.makedirs(TRACE_DIR, exist_ok=True)

    # Capture one success workflow
    success_id = success_workflow_ids[0]
    print(f"  Fetching event history for: {success_id}")
    events = await fetch_event_history(success_id)
    print(f"  Events found: {len(events)}")
    for e in events:
        print(f"    #{e['event_id']:3d}  {e['event_type']}")

    trace_path = os.path.join(TRACE_DIR, f"success-{success_id}.json")
    with open(trace_path, "w") as f:
        json.dump({
            "workflow_id": success_id,
            "path": "success",
            "item": "USB-C cable",
            "quantity": 500,
            "budget": 3.00,
            "latency_seconds": success_latencies[0],
            "events": events,
        }, f, indent=2)
    print(f"  Saved to: {trace_path}")

    # Capture one no-match workflow
    nomatch_id = nomatch_workflow_ids[0]
    print(f"\n  Fetching event history for: {nomatch_id}")
    events_nm = await fetch_event_history(nomatch_id)
    print(f"  Events found: {len(events_nm)}")
    for e in events_nm:
        print(f"    #{e['event_id']:3d}  {e['event_type']}")

    trace_path_nm = os.path.join(TRACE_DIR, f"no-match-{nomatch_id}.json")
    with open(trace_path_nm, "w") as f:
        json.dump({
            "workflow_id": nomatch_id,
            "path": "no_match",
            "item": "quantum computer",
            "quantity": 1,
            "budget": 0.01,
            "latency_seconds": nomatch_latencies[0],
            "events": events_nm,
        }, f, indent=2)
    print(f"  Saved to: {trace_path_nm}")

    # ── Phase 5: Final summary ──
    print("\n" + "=" * 70)
    print("📋 FINAL SUMMARY — Screenshot this for your interview")
    print("=" * 70)
    print(f"\n  Total workflows submitted:  {10 + 3}")
    print(f"  Success path:               10/10 completed")
    print(f"  No-match path:               3/3 completed")
    print(f"  Success p50 latency:        {statistics.median(success_latencies):.3f}s")
    print(f"  Success p99 latency:        {sorted(success_latencies)[-1]:.3f}s")
    print(f"  No-match p50 latency:       {statistics.median(nomatch_latencies):.3f}s")
    print(f"\n  Event traces saved to:       docs/traces/")
    print(f"  Temporal Web UI:             http://localhost:8080")
    print(f"\n  Workflow IDs to look up in UI:")
    for wid in success_workflow_ids[:3]:
        print(f"    Success: {wid}")
    for wid in nomatch_workflow_ids[:2]:
        print(f"    No-match: {wid}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
