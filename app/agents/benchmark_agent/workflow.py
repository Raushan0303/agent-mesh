"""BenchmarkWorkflow — minimal workflow for load testing.

Executes a single activity that sleeps for N ms and returns a payload.
No checkpointing, no interrupt, no tools, no LLM, no Postgres.
This isolates the Temporal infrastructure throughput from application logic.
"""

import asyncio
import time
from datetime import timedelta

from temporalio import activity, workflow


@activity.defn
async def benchmark_activity(sleep_ms: int, payload_size: int) -> dict:
    """Sleep for sleep_ms, return a payload of payload_size bytes."""
    start = time.monotonic()
    await asyncio.sleep(sleep_ms / 1000.0)
    elapsed_ms = (time.monotonic() - start) * 1000
    # Generate a dummy payload (just a string of the right size)
    payload = "x" * min(payload_size, 10000)
    return {
        "elapsed_ms": round(elapsed_ms, 2),
        "payload_len": len(payload),
        "worker_pid": __import__("os").getpid(),
    }


@workflow.defn
class BenchmarkWorkflow:
    """Minimal workflow: one activity, no signals, no checkpoints.

    Tests: gateway → Temporal dispatch → task queue → worker → activity → return.
    """

    @workflow.run
    async def run(self, data: dict) -> dict:
        sleep_ms = data.get("sleep_ms", 100)
        payload_size = data.get("payload_size", 100)

        result = await workflow.execute_activity(
            benchmark_activity,
            args=(sleep_ms, payload_size),
            start_to_close_timeout=timedelta(seconds=30),
            schedule_to_close_timeout=timedelta(seconds=60),
        )
        return result


@workflow.defn
class HoldWorkflow:
    """Holds a durable timer, then runs the benchmark activity.

    Used to measure how many workflows can be OPEN at the same time: a
    workflow waiting on a timer (or a human approval) is a row in
    Temporal's database, not a running coroutine on a worker.
    """

    @workflow.run
    async def run(self, data: dict) -> dict:
        await asyncio.sleep(data.get("hold_s", 60))  # durable timer in workflow code
        return await workflow.execute_activity(
            benchmark_activity,
            args=(data.get("sleep_ms", 100), data.get("payload_size", 100)),
            start_to_close_timeout=timedelta(seconds=30),
            schedule_to_close_timeout=timedelta(minutes=30),
        )
