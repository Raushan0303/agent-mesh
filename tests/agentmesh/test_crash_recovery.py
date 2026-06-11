"""
Automated crash recovery integration test.

Starts a worker, submits a workflow, kills the worker mid-execution,
starts a second worker, and asserts the workflow still completes.

Requires: Temporal cluster running (docker-compose up -d).
"""
import asyncio
import subprocess
import sys
import time

import pytest
from temporalio.client import Client

from app.core.config import settings


async def _wait_for_status(client: Client, workflow_id: str, target: str, timeout: float = 30):
    """Poll workflow status until it matches target or timeout."""
    handle = client.get_workflow_handle(workflow_id)
    deadline = time.time() + timeout
    while time.time() < deadline:
        desc = await handle.describe()
        if desc.status.name == target:
            return True
        await asyncio.sleep(0.5)
    return False


@pytest.mark.asyncio
async def test_crash_recovery_worker_killed_workflow_completes():
    """
    1. Start Worker A in a subprocess.
    2. Submit a workflow via Temporal client (bypass gateway for isolation).
    3. Kill Worker A immediately (mid-poll or mid-execution).
    4. Start Worker B in a subprocess.
    5. Send approval signal (Week 4: workflow pauses at Approve node).
    6. Assert the workflow reaches COMPLETED status.

    This proves Temporal redelivers the Activity Task to a new Worker
    after the original Worker dies — the core durability guarantee.
    """
    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
    )

    worker_a = subprocess.Popen(
        [sys.executable, "-m", "app.agentmesh.worker_runner"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    await asyncio.sleep(2)

    from app.agents.sourcing_agent.workflow import SourcingWorkflow
    from app.agents.sourcing_agent.state import SourcingBriefInput
    from app.agents.sourcing_agent import TASK_QUEUE

    workflow_id = f"crash-test-{int(time.time())}"
    await client.start_workflow(
        SourcingWorkflow.run,
        SourcingBriefInput(item="USB-C cable", quantity=500, budget=3.00),
        id=workflow_id,
        task_queue=TASK_QUEUE,
    )

    worker_a.kill()
    worker_a.wait(timeout=5)

    handle = client.get_workflow_handle(workflow_id)
    desc = await handle.describe()
    status_before = desc.status.name
    assert status_before in ("RUNNING", "COMPLETED"), f"Unexpected status: {status_before}"

    if status_before == "COMPLETED":
        return

    worker_b = subprocess.Popen(
        [sys.executable, "-m", "app.agentmesh.worker_runner"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        # Week 4: the workflow pauses at the Approve node waiting for a signal.
        # Wait a bit for the graph to reach the Approve node, then send approval.
        await asyncio.sleep(3)
        await handle.signal("approve", {"approved": True, "comment": "crash test auto-approve"})

        completed = await _wait_for_status(client, workflow_id, "COMPLETED", timeout=30)
        assert completed, "Workflow did not complete after Worker B started"
    finally:
        worker_b.terminate()
        worker_b.wait(timeout=5)
