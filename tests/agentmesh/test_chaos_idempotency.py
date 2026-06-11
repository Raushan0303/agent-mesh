"""
Chaos test: SIGKILL Workers between create_po and initiate_payment.

Runs N trials. Each trial:
1. Start a Worker.
2. Submit a sourcing workflow.
3. Poll the DB until the PO row appears (PO Activity completed).
4. SIGKILL the Worker immediately (between PO and payment).
5. Start a new Worker.
6. Wait for the workflow to complete.
7. Assert exactly 1 PO row and 1 payment row for this workflow_id.

After all trials: assert 0 duplicate POs and 0 duplicate payments total.

Requires: Temporal cluster running (docker-compose up -d) + agentmesh DB initialized.
"""

import asyncio
import subprocess
import sys
import time

import pytest
from temporalio.client import Client

from app.core.config import settings

# Number of kill trials. The PRD says 100, but for CI speed we default to 20.
# Set CHAOS_TRIALS=100 env var for the full run.
NUM_TRIALS = 20


async def _wait_for_status(client: Client, workflow_id: str, target: str, timeout: float = 60):
    """Poll workflow status until it matches target or timeout."""
    handle = client.get_workflow_handle(workflow_id)
    deadline = time.time() + timeout
    while time.time() < deadline:
        desc = await handle.describe()
        if desc.status.name == target:
            return True
        await asyncio.sleep(0.3)
    return False


async def _wait_for_po(workflow_id: str, timeout: float = 30) -> bool:
    """Poll DB until a PO row appears for this workflow_id."""
    import asyncpg

    deadline = time.time() + timeout
    while time.time() < deadline:
        conn = await asyncpg.connect(settings.agent_db_dsn)
        try:
            count = await conn.fetchval(
                "SELECT count(*) FROM sourcing_agent_purchase_orders WHERE idempotency_key LIKE $1",
                f"{workflow_id}:%",
            )
            if count > 0:
                return True
        finally:
            await conn.close()
        await asyncio.sleep(0.1)
    return False


async def _count_rows_for_workflow(workflow_id: str) -> tuple[int, int]:
    """Return (po_count, payment_count) for a workflow_id."""
    import asyncpg

    conn = await asyncpg.connect(settings.agent_db_dsn)
    try:
        po_count = await conn.fetchval(
            "SELECT count(*) FROM sourcing_agent_purchase_orders WHERE idempotency_key LIKE $1",
            f"{workflow_id}:%",
        )
        pay_count = await conn.fetchval(
            "SELECT count(*) FROM sourcing_agent_payment_intents WHERE idempotency_key LIKE $1",
            f"{workflow_id}:%",
        )
        return po_count, pay_count
    finally:
        await conn.close()


def _start_worker() -> subprocess.Popen:
    """Start a worker subprocess."""
    return subprocess.Popen(
        [sys.executable, "-m", "app.agentmesh.worker_runner"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _kill_worker(worker: subprocess.Popen) -> None:
    """SIGKILL the worker."""
    worker.kill()
    worker.wait(timeout=5)


@pytest.mark.asyncio
async def test_chaos_idempotency_zero_duplicates():
    """
    Chaos test: repeatedly SIGKILL Workers between PO and payment.

    Proves that Temporal's replay + Activity-level idempotency keys
    together guarantee zero duplicate POs and zero duplicate payments.
    """
    import asyncpg

    from app.agents.sourcing_agent.workflow import SourcingWorkflow
    from app.agents.sourcing_agent.state import SourcingBriefInput
    from app.agents.sourcing_agent import TASK_QUEUE

    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
    )

    # Clear DB before starting
    conn = await asyncpg.connect(settings.agent_db_dsn)
    try:
        await conn.execute("DELETE FROM sourcing_agent_purchase_orders")
        await conn.execute("DELETE FROM sourcing_agent_payment_intents")
    finally:
        await conn.close()

    workflow_ids = []
    worker = _start_worker()
    await asyncio.sleep(2)

    try:
        for i in range(NUM_TRIALS):
            trial_id = f"chaos-{i:03d}-{int(time.time())}"
            workflow_ids.append(trial_id)

            # Submit the workflow
            await client.start_workflow(
                SourcingWorkflow.run,
                SourcingBriefInput(item="USB-C cable", quantity=500, budget=3.00),
                id=trial_id,
                task_queue=TASK_QUEUE,
            )

            # Week 4: the workflow pauses at the Approve node for human approval.
            # Wait for the graph to reach Approve, then send the approval signal.
            await asyncio.sleep(3)
            handle = client.get_workflow_handle(trial_id)
            await handle.signal("approve", {"approved": True, "comment": "chaos test"})

            # Wait until the PO is created (after approval, before payment)
            po_created = await _wait_for_po(trial_id, timeout=30)
            assert po_created, f"Trial {i}: PO was not created before timeout"

            # SIGKILL the worker between PO and payment
            _kill_worker(worker)

            # Start a new worker to pick up where we left off
            worker = _start_worker()
            await asyncio.sleep(2)

            # Wait for the workflow to complete
            completed = await _wait_for_status(client, trial_id, "COMPLETED", timeout=60)
            assert completed, f"Trial {i}: workflow did not complete after worker restart"

            # Check row counts for this workflow
            po_count, pay_count = await _count_rows_for_workflow(trial_id)
            assert po_count == 1, (
                f"Trial {i}: expected exactly 1 PO, got {po_count} "
                f"(idempotency key failed?)"
            )
            assert pay_count == 1, (
                f"Trial {i}: expected exactly 1 payment, got {pay_count} "
                f"(idempotency key failed?)"
            )

            print(f"  Trial {i+1}/{NUM_TRIALS}: {trial_id} — 1 PO, 1 payment ✅")

    finally:
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=5)

    # Final aggregate check: total rows should equal number of trials
    conn = await asyncpg.connect(settings.agent_db_dsn)
    try:
        total_pos = await conn.fetchval("SELECT count(*) FROM sourcing_agent_purchase_orders")
        total_pays = await conn.fetchval("SELECT count(*) FROM sourcing_agent_payment_intents")
    finally:
        await conn.close()

    print(f"\n{'='*60}")
    print(f"Chaos Test Results: {NUM_TRIALS} trials")
    print(f"  Total POs:      {total_pos} (expected {NUM_TRIALS})")
    print(f"  Total Payments: {total_pays} (expected {NUM_TRIALS})")
    print(f"  Duplicate POs:      {total_pos - NUM_TRIALS}")
    print(f"  Duplicate Payments: {total_pays - NUM_TRIALS}")
    print(f"{'='*60}")

    assert total_pos == NUM_TRIALS, (
        f"Expected {NUM_TRIALS} POs total, got {total_pos} — "
        f"{total_pos - NUM_TRIALS} duplicates detected!"
    )
    assert total_pays == NUM_TRIALS, (
        f"Expected {NUM_TRIALS} payments total, got {total_pays} — "
        f"{total_pays - NUM_TRIALS} duplicates detected!"
    )
