"""Benchmark agent — minimal workflow for load testing.

No Postgres, no LLM, no memory, no tools. Just sleep + return.
This tests the raw infrastructure: gateway → Temporal → task queue → worker.
"""

TASK_QUEUE = "benchmark-task-queue"

from pydantic import BaseModel

from app.agentmesh.gateway.registry import AGENT_REGISTRY, AgentRegistration
from app.agents.benchmark_agent.workflow import BenchmarkWorkflow


class BenchmarkInput(BaseModel):
    """Input for the benchmark workflow — just a duration and payload size."""
    sleep_ms: int = 100
    payload_size: int = 100


AGENT_REGISTRY["benchmark_agent"] = AgentRegistration(
    workflow_class=BenchmarkWorkflow,
    task_queue=TASK_QUEUE,
    input_model=BenchmarkInput,
)
