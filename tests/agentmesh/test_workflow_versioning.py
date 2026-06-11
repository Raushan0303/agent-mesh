"""
Workflow Versioning test (Week 3, updated for Week 4).

The Week 4 workflow pauses at the Approve node for human approval.
This test sends the approval signal and verifies the workflow completes
with the full pipeline (graph → PO → payment).
"""

import asyncio
import subprocess
import sys
import time

import pytest
from temporalio.client import Client

from app.core.config import settings


async def _wait_for_completion(client: Client, workflow_id: str, timeout: float = 60) -> bool:
    handle = client.get_workflow_handle(workflow_id)
    deadline = time.time() + timeout
    while time.time() < deadline:
        desc = await handle.describe()
        if desc.status.name == "COMPLETED":
            return True
        await asyncio.sleep(0.3)
    return False


def _count_activities(history) -> int:
    """Count ActivityTaskCompleted events in the history."""
    count = 0
    for event in history.events:
        if event.event_type:
            etype = int(event.event_type)
            if etype == 12:  # ActivityTaskCompleted
                count += 1
    return count


@pytest.mark.asyncio
async def test_workflow_versioning_new_workflow_uses_scored_path():
    """
    1. Start a Worker.
    2. Submit a sourcing workflow.
    3. Wait for the graph to pause at Approve.
    4. Send approval signal.
    5. Assert the workflow completes successfully with PO + payment.
    6. Assert multiple Activities ran (graph + resume + PO + payment).
    """
    from app.agents.sourcing_agent.workflow import SourcingWorkflow
    from app.agents.sourcing_agent.state import SourcingBriefInput
    from app.agents.sourcing_agent import TASK_QUEUE

    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
    )

    worker = subprocess.Popen(
        [sys.executable, "-m", "app.agentmesh.worker_runner"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    await asyncio.sleep(2)

    try:
        workflow_id = f"versioning-test-{int(time.time())}"
        await client.start_workflow(
            SourcingWorkflow.run,
            SourcingBriefInput(item="USB-C cable", quantity=500, budget=3.00),
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )

        # Wait for the graph to reach the Approve node (give it time to run)
        await asyncio.sleep(3)

        # Send approval signal
        handle = client.get_workflow_handle(workflow_id)
        await handle.signal("approve", {"approved": True, "comment": "versioning test"})

        completed = await _wait_for_completion(client, workflow_id, timeout=60)
        assert completed, "Workflow did not complete"

        # Fetch the result and event history
        result = await handle.result()
        history = await handle.fetch_history()

        if hasattr(result, "model_dump"):
            result = result.model_dump()
        elif not isinstance(result, dict):
            result = dict(result)

        activity_count = _count_activities(history)
        print(f"\n  Activities completed: {activity_count}")
        print(f"  Result status: {result.get('status')}")
        print(f"  Result po_id: {result.get('po_id')}")
        print(f"  Result payment_id: {result.get('payment_id')}")
        print(f"  Approval status: {result.get('approval_status')}")
        print(f"  Selected supplier: {result.get('selected_supplier')}")

        # Week 4: graph runs as 2 Activities (until_interrupt + resume),
        # plus create_po + initiate_payment = 4 total
        assert activity_count >= 3, (
            f"Expected at least 3 activities, got {activity_count}."
        )

        assert result.get("status") == "completed"
        assert result.get("po_id") is not None, "PO should have been created"
        assert result.get("payment_id") is not None, "Payment should have been initiated"
        assert result.get("approval_status") == "approved"

        print("  Versioning test PASSED ✅ — workflow completed with approval")

    finally:
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=5)
