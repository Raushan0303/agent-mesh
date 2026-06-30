"""Locust load test for AgentMesh benchmark workflows.

Usage:
    locust -f scripts/locustfile.py --host http://localhost:8000

Then open http://localhost:8089 in your browser.
Set Number of users = 100, Ramp up = 10, and start.

Each "user" hits POST /workflows/benchmark with count=1, which launches
one real Temporal workflow and waits for it to complete.
This tests end-to-end throughput: gateway → Temporal → worker → activity → return.
"""

import json
import random

from locust import HttpUser, task, between


class WorkflowLoadTestUser(HttpUser):
    """Simulates a client launching benchmark workflows."""

    wait_time = between(0.1, 0.5)

    @task
    def launch_benchmark_workflow(self):
        self.client.post(
            "/workflows/benchmark",
            json={
                "count": 1,
                "sleep_ms": random.randint(50, 200),
                "payload_size": 100,
                "concurrency": 1,
            },
            name="/workflows/benchmark",
        )

    @task(3)
    def launch_batch_workflows(self):
        """Launch 10 workflows at once to test batch throughput."""
        self.client.post(
            "/workflows/benchmark",
            json={
                "count": 10,
                "sleep_ms": 100,
                "payload_size": 100,
                "concurrency": 10,
            },
            name="/workflows/benchmark [batch=10]",
        )

    @task(1)
    def launch_large_batch(self):
        """Launch 100 workflows at once — stress test."""
        self.client.post(
            "/workflows/benchmark",
            json={
                "count": 100,
                "sleep_ms": 50,
                "payload_size": 100,
                "concurrency": 50,
            },
            name="/workflows/benchmark [batch=100]",
        )
