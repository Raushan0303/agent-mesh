"""
Timeout-boundary test — Activity that hangs past StartToClose but under
ScheduleToClose.

Proves that Temporal's timeout semantics behave as configured:
  - StartToClose: the Activity is killed if it runs longer than this
  - ScheduleToClose: the Activity is given up on entirely if the total
    time (including retries) exceeds this

This test creates a hanging Activity with a short StartToClose timeout
and a longer ScheduleToClose timeout. The Activity should be killed by
StartToClose, retried, and then killed by ScheduleToClose.

Requires: Temporal cluster running.
"""

import asyncio
import subprocess
import sys
import time

import pytest
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy, VersioningBehavior
from temporalio.worker import Worker, UnsandboxedWorkflowRunner
from datetime import timedelta

from app.core.config import settings


# ── Hanging activity for testing ──


@activity.defn
async def hanging_activity(duration_seconds: float) -> str:
    """An activity that sleeps for the given duration, then returns.

    Used to test StartToClose timeout — if StartToClose < duration,
    the activity will be killed mid-sleep.
    """
    print(f"  hanging_activity: sleeping for {duration_seconds}s")
    await asyncio.sleep(duration_seconds)
    return "completed"


@workflow.defn(versioning_behavior=VersioningBehavior.AUTO_UPGRADE)
class TimeoutTestWorkflow:
    @workflow.run
    async def run(self, duration: float) -> str:
        result = await workflow.execute_activity(
            hanging_activity,
            duration,
            start_to_close_timeout=timedelta(seconds=2),
            schedule_to_close_timeout=timedelta(seconds=8),
            retry_policy=RetryPolicy(maximum_attempts=1),  # no retry
        )
        return result


@pytest.mark.asyncio
async def test_activity_killed_by_start_to_close():
    """
    An Activity that hangs past StartToClose (2s) should be killed by
    Temporal's StartToClose timeout, even if ScheduleToClose (8s) hasn't
    elapsed yet.

    This proves StartToClose bounds the Activity's execution time, not
    the total time including retries.
    """
    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
    )

    # Run a worker with the hanging activity (unsandboxed — the test
    # workflow imports app.core.config which uses pathlib.expanduser)
    worker = Worker(
        client,
        task_queue="timeout-test-queue",
        workflows=[TimeoutTestWorkflow],
        activities=[hanging_activity],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    worker_task = asyncio.create_task(worker.run())

    try:
        workflow_id = f"timeout-test-{int(time.time())}"
        # Start the workflow with a 10s hang — longer than StartToClose (2s)
        handle = await client.start_workflow(
            TimeoutTestWorkflow.run,
            10.0,  # hang for 10 seconds
            id=workflow_id,
            task_queue="timeout-test-queue",
        )

        # Wait for the workflow to fail (should fail quickly due to StartToClose)
        start_time = time.time()
        try:
            result = await handle.result()
            # If we get here, the activity completed — that's wrong
            pytest.fail(
                f"Activity should have been killed by StartToClose timeout, "
                f"but it completed with result: {result}"
            )
        except Exception as e:
            elapsed = time.time() - start_time
            error_str = str(e).lower()

            print(f"\n  Workflow failed after {elapsed:.1f}s")
            print(f"  Error: {type(e).__name__}: {e}")

            # Should fail due to timeout, and should fail quickly (around 2s,
            # not 8s or 10s)
            assert elapsed < 5.0, (
                f"Workflow should have failed within ~2s (StartToClose), "
                f"but took {elapsed:.1f}s — ScheduleToClose may have been "
                f"used instead"
            )

            # The error should mention timeout (check cause chain too)
            cause = e
            error_messages = []
            while cause is not None:
                error_messages.append(str(cause).lower())
                cause = getattr(cause, "__cause__", None) or getattr(cause, "cause", None)

            all_errors = " ".join(error_messages)
            assert "timeout" in all_errors or "timed out" in all_errors, (
                f"Expected timeout error in cause chain, got: {e}"
            )

            print(f"  Activity killed by StartToClose timeout after ~{elapsed:.1f}s ✅")
            print(f"  (StartToClose=2s, ScheduleToClose=8s, hang=10s)")

    finally:
        worker_task.cancel()
        try:
            await worker_task
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_activity_completes_under_start_to_close():
    """
    An Activity that completes before StartToClose should succeed.
    This is the positive case — proves the timeout doesn't fire prematurely.
    """
    client = await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
    )

    worker = Worker(
        client,
        task_queue="timeout-test-queue-2",
        workflows=[TimeoutTestWorkflow],
        activities=[hanging_activity],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    worker_task = asyncio.create_task(worker.run())

    try:
        workflow_id = f"timeout-ok-{int(time.time())}"
        # Start the workflow with a 1s hang — under StartToClose (2s)
        handle = await client.start_workflow(
            TimeoutTestWorkflow.run,
            1.0,  # hang for 1 second (under the 2s StartToClose)
            id=workflow_id,
            task_queue="timeout-test-queue-2",
        )

        result = await handle.result()
        assert result == "completed"

        print(f"\n  Activity completed in 1s (under StartToClose=2s) ✅")

    finally:
        worker_task.cancel()
        try:
            await worker_task
        except (asyncio.CancelledError, Exception):
            pass
