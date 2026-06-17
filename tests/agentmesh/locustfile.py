"""Locust load test — simulates 500 concurrent sourcing briefs.

Reports p50/p95/p99 latency and throughput against the Temporal cluster.

Usage:
    locust -f tests/agentmesh/locustfile.py --headless \
        --users 500 --spawn-rate 10 --run-time 60s \
        --host http://localhost:8000

Or with the web UI:
    locust -f tests/agentmesh/locustfile.py --host http://localhost:8000
"""

import json
import random
import time

from locust import HttpUser, task, between


ITEMS = [
    ("USB-C cable", 3.00, 500),
    ("HDMI cable", 5.00, 200),
    ("Ethernet cable", 2.00, 300),
    ("Power adapter", 10.00, 100),
    ("USB-C cable", 3.50, 1000),
    ("HDMI cable", 4.50, 50),
    ("Ethernet cable", 1.50, 1000),
]


class SourcingAgentUser(HttpUser):
    """Simulates a founder submitting sourcing briefs."""

    wait_time = between(1, 3)  # wait 1-3 seconds between requests

    def on_start(self):
        """Called when a user starts — track pending workflows."""
        self.pending_workflows = []

    @task(3)
    def submit_sourcing_brief(self):
        """Submit a sourcing brief and track the workflow."""
        item, budget, quantity = random.choice(ITEMS)

        response = self.client.post(
            "/workflows",
            json={
                "agent_type": "sourcing_agent",
                "input": {
                    "item": item,
                    "quantity": quantity,
                    "budget": budget,
                },
            },
            name="POST /workflows (submit brief)",
        )

        if response.status_code == 200:
            data = response.json()
            wf_id = data.get("workflow_id")
            if wf_id:
                self.pending_workflows.append(wf_id)

    @task(1)
    def check_workflow_status(self):
        """Check the status of a pending workflow."""
        if not self.pending_workflows:
            return

        wf_id = self.pending_workflows.pop(0)
        self.client.get(
            f"/workflows/{wf_id}",
            name="GET /workflows/{id} (status check)",
        )

    @task(1)
    def approve_workflow(self):
        """Approve a pending workflow (simulates human approval)."""
        if not self.pending_workflows:
            return

        # Wait a bit for the workflow to reach the Approve node
        time.sleep(2)

        wf_id = self.pending_workflows.pop(0)
        self.client.post(
            f"/workflows/{wf_id}/approve",
            json={"approved": True, "comment": "load test auto-approve"},
            name="POST /workflows/{id}/approve",
        )
