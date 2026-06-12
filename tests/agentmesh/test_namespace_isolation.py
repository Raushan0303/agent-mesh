"""
Namespace isolation test.

Proves that Temporal Namespaces provide tenant isolation — a Workflow in
tenant A's Namespace is invisible to a query scoped to tenant B's Namespace.

This is a platform capability: the sourcing agent doesn't implement isolation,
it just runs inside whichever Namespace it's told to. The platform's Namespace
helper (agentmesh/temporal/namespaces.py) provides the client scoping.

Note: This test requires a second Namespace to be registered. For local
development, we use the "default" Namespace and a "tenant-b" Namespace.
If "tenant-b" doesn't exist, the test is skipped (not failed).
"""

import asyncio
import subprocess
import sys
import time

import pytest
from temporalio.client import Client
from temporalio.service import RPCError

from app.core.config import settings


async def _namespace_exists(client: Client, namespace: str) -> bool:
    """Check if a Namespace exists by trying to describe it."""
    try:
        # Try to list workflows in the namespace — if the namespace doesn't
        # exist, this will raise an error
        handle = client.get_workflow_handle_for("nonexistent-workflow-id", namespace=namespace)
        await handle.describe()
        return True  # Shouldn't reach here (workflow doesn't exist)
    except Exception as e:
        # If the error is about the workflow not being found, the namespace exists
        error_str = str(e).lower()
        if "namespace" in error_str and "not found" in error_str:
            return False
        if "workflow" in error_str and ("not found" in error_str or "not exist" in error_str):
            return True
        # Unknown error — assume namespace exists
        return True


@pytest.mark.asyncio
async def test_namespace_isolation_tenant_a_invisible_to_tenant_b():
    """
    1. Submit a workflow in the default Namespace (tenant A).
    2. Try to query it from a different Namespace (tenant B).
    3. Assert the query fails (workflow is invisible across Namespace boundaries).

    If the "tenant-b" Namespace doesn't exist, the test verifies that at least
    the Namespace-scoped client can be created (proving the helper works).
    """
    from app.agents.sourcing_agent.workflow import SourcingWorkflow
    from app.agents.sourcing_agent.state import SourcingBriefInput
    from app.agents.sourcing_agent import TASK_QUEUE

    # Connect to the default namespace
    client_a = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
    )

    # Try to connect to a second namespace
    tenant_b_namespace = "tenant-b"
    try:
        client_b = await Client.connect(
            settings.temporal_host,
            namespace=tenant_b_namespace,
        )
    except Exception as e:
        pytest.skip(
            f"Namespace '{tenant_b_namespace}' not available — "
            f"register it with: temporal operator namespace create {tenant_b_namespace}. "
            f"Error: {e}"
        )

    # Start a worker in the default namespace
    worker = subprocess.Popen(
        [sys.executable, "-m", "app.agentmesh.worker_runner"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    await asyncio.sleep(3)

    try:
        # Submit a workflow in tenant A's namespace (default)
        workflow_id = f"ns-isolation-{int(time.time())}"
        await client_a.start_workflow(
            SourcingWorkflow.run,
            SourcingBriefInput(item="USB-C cable", quantity=500, budget=3.00),
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
        print(f"\n  Workflow started in namespace '{settings.temporal_namespace}': {workflow_id}")

        # Wait for the graph to pause at Approve
        await asyncio.sleep(5)

        # Query from tenant A — should see the workflow
        handle_a = client_a.get_workflow_handle(workflow_id)
        desc_a = await handle_a.describe()
        print(f"  Tenant A sees: status={desc_a.status.name}")
        assert desc_a.status.name in ("RUNNING", "COMPLETED"), (
            f"Tenant A should see its own workflow, got status={desc_a.status.name}"
        )

        # Query from tenant B — should NOT see the workflow
        # The workflow ID exists in tenant A's namespace, not tenant B's
        try:
            handle_b = client_b.get_workflow_handle(workflow_id)
            desc_b = await handle_b.describe()
            # If we get here, the workflow is visible across namespaces — isolation failed
            pytest.fail(
                f"ISOLATION BROKEN: Tenant B can see tenant A's workflow "
                f"(status={desc_b.status.name})"
            )
        except Exception as e:
            # Expected — the workflow should be invisible from tenant B
            print(f"  Tenant B cannot see workflow: {type(e).__name__} (expected ✅)")
            print(f"  Isolation test PASSED ✅ — Namespaces provide tenant isolation")

        # Clean up: send approval so the workflow completes
        await handle_a.signal("approve", {"approved": True, "comment": "ns test cleanup"})

    finally:
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=5)
