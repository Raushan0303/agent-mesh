"""Benchmark route — launches N concurrent benchmark workflows and collects results."""

import asyncio
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.agentmesh.temporal.client import get_temporal_client

router = APIRouter(prefix="/workflows", tags=["benchmark"])


@router.post("/benchmark")
async def run_benchmark(request: Request) -> JSONResponse:
    """Launch N concurrent benchmark workflows and collect results.

    Request body:
        {
            "count": 100,          # number of workflows to launch
            "sleep_ms": 100,       # activity sleep duration
            "payload_size": 100,   # dummy payload size
            "concurrency": 50,     # how many to launch in parallel
            "workers": 1           # how many workers to spin up for this test
        }

    Returns:
        {
            "total": 100,
            "completed": 98,
            "failed": 2,
            "throughput": 45.2,       # workflows/sec
            "p50_ms": 120,
            "p99_ms": 340,
            "duration_ms": 2210,
            "workers": 1
        }
    """
    data = await request.json()
    count = data.get("count", 100)
    sleep_ms = data.get("sleep_ms", 100)
    payload_size = data.get("payload_size", 100)
    concurrency = data.get("concurrency", 50)
    num_workers = data.get("workers", 1)

    client = await get_temporal_client()
    from app.agents.benchmark_agent.workflow import BenchmarkWorkflow
    from app.agents.benchmark_agent import TASK_QUEUE as BENCHMARK_TASK_QUEUE

    # Spawn additional benchmark workers as subprocesses
    # The main worker_runner already has 1 benchmark worker running.
    # We spawn (num_workers - 1) additional workers as separate processes.
    import subprocess
    import sys
    extra_procs = []
    for i in range(num_workers - 1):
        proc = subprocess.Popen(
            [sys.executable, "-c", """
import asyncio
from app.agentmesh.temporal.client import get_temporal_client
from app.agentmesh.temporal.worker import run_worker
from app.agents.benchmark_agent.workflow import BenchmarkWorkflow, benchmark_activity
from app.agents.benchmark_agent import TASK_QUEUE

async def main():
    client = await get_temporal_client()
    await run_worker(
        client=client,
        task_queue=TASK_QUEUE,
        workflows=[BenchmarkWorkflow],
        activities=[benchmark_activity],
        max_concurrent_activities=100,
    )

asyncio.run(main())
"""],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        extra_procs.append(proc)

    # Give extra workers a moment to start polling
    if extra_procs:
        await asyncio.sleep(2)

    # Launch workflows with bounded concurrency
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    latencies = []
    spans = []  # (start, end) per completed workflow → measured concurrency
    errors = []
    batch_id = uuid.uuid4().hex[:8]

    async def launch_one(i: int):
        async with semaphore:
            wf_id = f"benchmark-{batch_id}-{i:05d}"
            start = time.monotonic()
            try:
                handle = await client.start_workflow(
                    BenchmarkWorkflow.run,
                    {"sleep_ms": sleep_ms, "payload_size": payload_size},
                    id=wf_id,
                    task_queue=BENCHMARK_TASK_QUEUE,
                )
                result = await handle.result()
                end = time.monotonic()
                elapsed_ms = (end - start) * 1000
                latencies.append(elapsed_ms)
                spans.append((start, end))
                results.append({"id": wf_id, "status": "completed", "ms": round(elapsed_ms, 1)})
            except Exception as e:
                elapsed_ms = (time.monotonic() - start) * 1000
                errors.append({"id": wf_id, "error": str(e), "ms": round(elapsed_ms, 1)})

    overall_start = time.monotonic()
    await asyncio.gather(*[launch_one(i) for i in range(count)])
    overall_ms = (time.monotonic() - overall_start) * 1000

    # Kill extra worker subprocesses — they were only for this test
    for proc in extra_procs:
        proc.terminate()
    for proc in extra_procs:
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    # Calculate stats
    completed = len(results)
    failed = len(errors)
    throughput = round(completed / (overall_ms / 1000), 1) if overall_ms > 0 else 0

    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) > 0 else 0
    avg = sum(latencies) / len(latencies) if latencies else 0

    # Measured, not assumed: how many workflows were actually in flight.
    events = sorted([(a, 1) for a, _ in spans] + [(b, -1) for _, b in spans])
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    avg_in_flight = throughput * avg / 1000  # Little's Law: L = λ·W

    return JSONResponse({
        "batch_id": batch_id,
        "total": count,
        "completed": completed,
        "failed": failed,
        "throughput_per_sec": throughput,
        "p50_ms": round(p50, 1),
        "p99_ms": round(p99, 1),
        "avg_ms": round(avg, 1),
        "duration_ms": round(overall_ms, 1),
        "concurrency": concurrency,  # client-side cap, not a measurement
        "measured_peak_in_flight": peak,
        "measured_avg_in_flight": round(avg_in_flight, 1),
        "sleep_ms": sleep_ms,
        "workers": num_workers,
        "errors": errors[:5],  # first 5 errors if any
    })
