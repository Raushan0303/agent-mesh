"""Temporal load test that reports what it actually measured.

Two modes:

  throughput  N BenchmarkWorkflows (one 100 ms activity each), at most C in
              flight from the client, served by W worker processes.
  hold        N HoldWorkflows that each wait on a durable timer for H
              seconds before their activity: all N are started first, and
              the number OPEN at the same time is counted from Temporal's
              visibility store, not inferred.

Every run uses its own task queue and its own W freshly spawned worker
processes, so no stray worker can serve it. Per-workflow timings and the
summary are saved to benchmarks/results/<run_id>.json.

Concurrency is MEASURED, not assumed: from the per-workflow start/end
timestamps we compute the peak and the time-averaged number in flight, and
check Little's Law (L = throughput x mean latency).

Usage:
  PYTHONPATH=. venv/bin/python scripts/loadtest.py throughput --workflows 2000 --concurrency 200 --workers 1
  PYTHONPATH=. venv/bin/python scripts/loadtest.py hold --workflows 10000 --workers 4 --hold-s 120
"""
import argparse
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import time
import uuid

from temporalio.client import Client

from app.agents.benchmark_agent.workflow import BenchmarkWorkflow, HoldWorkflow
from app.core.config import settings

OUT = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "results"

WORKER_CODE = """
import asyncio, sys
from temporalio.client import Client
from temporalio.worker import Worker
from app.agents.benchmark_agent.workflow import BenchmarkWorkflow, HoldWorkflow, benchmark_activity
from app.core.config import settings

async def main():
    client = await Client.connect(settings.temporal_host, namespace=settings.temporal_namespace)
    await Worker(client, task_queue=sys.argv[1], workflows=[BenchmarkWorkflow, HoldWorkflow],
                 activities=[benchmark_activity], max_concurrent_activities=100,
                 max_concurrent_workflow_tasks=100).run()

asyncio.run(main())
"""


def spawn_workers(n, queue):
    env = {**os.environ, "PYTHONPATH": str(OUT.parent.parent)}
    return [subprocess.Popen([sys.executable, "-c", WORKER_CODE, queue], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(n)]


def stop_workers(procs):
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0


def concurrency_stats(spans, t0, t1):
    """Peak and time-averaged number of workflows in flight from (start, end) spans."""
    events = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans])
    cur = peak = 0
    area, last = 0.0, t0
    for t, d in events:
        area += cur * (t - last)
        last = t
        cur += d
        peak = max(peak, cur)
    return peak, area / (t1 - t0) if t1 > t0 else 0


async def run_throughput(client, args, queue, record):
    sem = asyncio.Semaphore(args.concurrency)
    spans, errors = [], []

    async def one(i):
        async with sem:
            s = time.monotonic()
            try:
                h = await client.start_workflow(BenchmarkWorkflow.run, {"sleep_ms": args.sleep_ms, "payload_size": 100},
                                                id=f"{queue}-{i:06d}", task_queue=queue)
                await h.result()
                spans.append((s, time.monotonic()))
            except Exception as e:  # noqa: BLE001
                errors.append(str(e)[:200])

    t0 = time.monotonic()
    await asyncio.gather(*(one(i) for i in range(args.workflows)))
    t1 = time.monotonic()
    lat = [(e - s) * 1000 for s, e in spans]
    peak, avg_in_flight = concurrency_stats(spans, t0, t1)
    thr = len(spans) / (t1 - t0)
    record.update({
        "completed": len(spans), "failed": len(errors), "errors_sample": errors[:5],
        "duration_s": round(t1 - t0, 2),
        "throughput_per_s": round(thr, 1),
        "latency_ms": {"p50": round(pct(lat, .5), 1), "p95": round(pct(lat, .95), 1),
                       "p99": round(pct(lat, .99), 1), "mean": round(sum(lat) / max(1, len(lat)), 1)},
        "measured_in_flight": {"peak": peak, "time_avg": round(avg_in_flight, 1)},
        "littles_law_L_equals_lambda_W": round(thr * (sum(lat) / max(1, len(lat))) / 1000, 1),
    })
    record["raw_spans_ms"] = [[round((s - t0) * 1000, 1), round((e - t0) * 1000, 1)] for s, e in spans]


async def count(client, query):
    return (await client.count_workflows(query)).count


async def run_hold(client, args, queue, record):
    q_open = f'WorkflowType="HoldWorkflow" AND TaskQueue="{queue}" AND ExecutionStatus="Running"'
    q_done = f'WorkflowType="HoldWorkflow" AND TaskQueue="{queue}" AND ExecutionStatus="Completed"'
    sem = asyncio.Semaphore(500)  # client-side cap on concurrent START calls only
    started, errors = 0, []

    async def start(i):
        nonlocal started
        async with sem:
            try:
                await client.start_workflow(HoldWorkflow.run, {"hold_s": args.hold_s, "sleep_ms": args.sleep_ms},
                                            id=f"{queue}-{i:06d}", task_queue=queue)
                started += 1
            except Exception as e:  # noqa: BLE001
                errors.append(str(e)[:200])

    t0 = time.monotonic()
    await asyncio.gather(*(start(i) for i in range(args.workflows)))
    t_started = time.monotonic() - t0
    samples, peak_open = [], 0
    deadline = time.monotonic() + args.hold_s + args.drain_timeout_s
    while time.monotonic() < deadline:
        o, d = await count(client, q_open), await count(client, q_done)
        samples.append({"t_s": round(time.monotonic() - t0, 1), "open": o, "completed": d})
        peak_open = max(peak_open, o)
        print(f"  t={samples[-1]['t_s']:>6}s open={o:>6} completed={d:>6}", flush=True)
        if d + len(errors) >= args.workflows or (o == 0 and d > 0):
            break
        await asyncio.sleep(5)
    record.update({
        "started": started, "start_errors": len(errors), "errors_sample": errors[:5],
        "start_phase_s": round(t_started, 1), "start_rate_per_s": round(started / t_started, 1),
        "peak_open_workflows_from_visibility": peak_open,
        "completed": samples[-1]["completed"] if samples else 0,
        "visibility_samples": samples,
    })


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["throughput", "hold"])
    p.add_argument("--workflows", type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--sleep-ms", type=int, default=100)
    p.add_argument("--hold-s", type=int, default=60)
    p.add_argument("--drain-timeout-s", type=int, default=900)
    args = p.parse_args()

    run_id = f"{args.mode}-{args.workflows}wf-{args.workers}w-{uuid.uuid4().hex[:6]}"
    queue = f"loadtest-{run_id}"
    client = await Client.connect(settings.temporal_host, namespace=settings.temporal_namespace)
    procs = spawn_workers(args.workers, queue)
    await asyncio.sleep(3)  # let workers start polling
    record = {"run_id": run_id, "mode": args.mode, "task_queue": queue, "workers": args.workers,
              "workflows": args.workflows, "client_concurrency_cap": args.concurrency if args.mode == "throughput" else None,
              "activity_sleep_ms": args.sleep_ms, "hold_s": args.hold_s if args.mode == "hold" else None,
              "machine": {"cpus": os.cpu_count()}, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        if args.mode == "throughput":
            await run_throughput(client, args, queue, record)
        else:
            await run_hold(client, args, queue, record)
    finally:
        stop_workers(procs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{run_id}.json").write_text(json.dumps(record, indent=1))
    summary = {k: v for k, v in record.items() if k not in ("raw_spans_ms", "visibility_samples")}
    print(json.dumps(summary, indent=2))
    print(f"saved benchmarks/results/{run_id}.json")


if __name__ == "__main__":
    asyncio.run(main())
