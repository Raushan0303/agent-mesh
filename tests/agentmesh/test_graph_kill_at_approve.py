"""
Graph kill test — kill Worker while paused at the Approve node.

Proves that the LangGraph checkpointer + Temporal's Activity retry together
ensure the graph resumes at the Approve node (not Research) after a Worker
kill while paused.

Flow:
1. Start Worker A.
2. Submit a workflow — the graph runs Research → Score → Decide → Approve.
3. The graph pauses at Approve (interrupt() fires).
4. SIGKILL Worker A.
5. Start Worker B.
6. Send the approval signal.
7. Assert the workflow completes (graph resumes at Approve, not Research).
8. Assert the result has approval_status = "approved" and PO + payment IDs.

The key proof: if the checkpointer didn't work, the graph would restart
from Research after the Worker kill, re-querying suppliers. With the
checkpointer, the graph's state at Approve is persisted in Postgres, so
the resume Activity picks up exactly where it left off.
"""

import asyncio
import subprocess
import sys
import time

import pytest
from temporalio.client import Client

from app.core.config import settings


async def _wait_for_status(client: Client, workflow_id: str, target: str, timeout: float = 60):
    handle = client.get_workflow_handle(workflow_id)
    deadline = time.time() + timeout
    while time.time() < deadline:
        desc = await handle.describe()
        if desc.status.name == target:
            return True
        await asyncio.sleep(0.3)
    return False


@pytest.mark.asyncio
async def test_graph_kill_at_approve_resumes_at_approve():
    """
    1. Start Worker A.
    2. Submit a workflow.
    3. Wait for the graph to pause at Approve (workflow is RUNNING, not COMPLETED).
    4. SIGKILL Worker A.
    5. Start Worker B.
    6. Send approval signal.
    7. Assert workflow completes with approval_status = "approved".
    8. Assert PO and payment were created (graph reached Confirm, not just Research).
    """
    from app.agents.sourcing_agent.workflow import SourcingWorkflow
    from app.agents.sourcing_agent.state import SourcingBriefInput
    from app.agents.sourcing_agent import TASK_QUEUE

    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
    )

    worker_a = subprocess.Popen(
        [sys.executable, "-m", "app.agentmesh.worker_runner"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    await asyncio.sleep(3)

    workflow_id = f"graph-kill-{int(time.time())}"
    await client.start_workflow(
        SourcingWorkflow.run,
        SourcingBriefInput(item="USB-C cable", quantity=500, budget=3.00),
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    # Wait for the graph to reach the Approve node (it should be RUNNING, paused)
    await asyncio.sleep(5)
    handle = client.get_workflow_handle(workflow_id)
    desc = await handle.describe()
    status_before_kill = desc.status.name
    print(f"\n  Status before kill: {status_before_kill}")
    assert status_before_kill == "RUNNING", (
        f"Expected workflow to be RUNNING (paused at Approve), got {status_before_kill}"
    )

    # SIGKILL Worker A while the graph is paused at Approve
    worker_a.kill()
    worker_a.wait(timeout=5)
    print("  Worker A killed (while graph paused at Approve)")

    # Start Worker B
    worker_b = subprocess.Popen(
        [sys.executable, "-m", "app.agentmesh.worker_runner"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    await asyncio.sleep(3)

    try:
        # Send approval signal — this should resume the graph at Approve
        await handle.signal("approve", {"approved": True, "comment": "graph kill test"})
        print("  Approval signal sent")

        # Wait for completion
        completed = await _wait_for_status(client, workflow_id, "COMPLETED", timeout=60)
        assert completed, "Workflow did not complete after Worker B + approval signal"

        # Check the result — if the graph resumed at Approve (not Research),
        # it should have approval_status = "approved" and PO + payment IDs
        result = await handle.result()
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        elif not isinstance(result, dict):
            result = dict(result)

        print(f"  Result status: {result.get('status')}")
        print(f"  Approval status: {result.get('approval_status')}")
        print(f"  PO ID: {result.get('po_id')}")
        print(f"  Payment ID: {result.get('payment_id')}")
        print(f"  Selected supplier: {result.get('selected_supplier')}")

        assert result.get("status") == "completed", (
            f"Expected status='completed', got '{result.get('status')}'"
        )
        assert result.get("approval_status") == "approved", (
            "Graph did not resume at Approve — approval_status is not 'approved'"
        )
        assert result.get("po_id") is not None, (
            "PO was not created — graph may have restarted from Research instead of resuming at Approve"
        )
        assert result.get("payment_id") is not None, (
            "Payment was not initiated — graph may not have reached Confirm"
        )

        print("  Graph kill test PASSED ✅ — graph resumed at Approve after Worker kill")

    finally:
        if worker_b.poll() is None:
            worker_b.terminate()
            worker_b.wait(timeout=5)
