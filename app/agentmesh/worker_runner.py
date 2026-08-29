import asyncio

from app.agentmesh.temporal.client import get_temporal_client
from app.agentmesh.temporal.worker import run_worker
from app.agentmesh.observability.tracing import setup_tracing

# Initialize OpenTelemetry tracing in the worker process.
# Temporal's TracingInterceptor (on the client) extracts the trace context
# from Temporal headers and activates it in the worker. When the decide_node
# calls the LLM client, the traceparent header is auto-injected from the
# active OTel context — no manual trace_id passing needed.
setup_tracing(service_name="agentmesh-worker")

# This is where the agent gets wired in — the ONLY place that imports agent code
from app.agents.sourcing_agent.workflow import SourcingWorkflow
from app.agents.sourcing_agent.activity import (
    create_po_activity,
    initiate_payment_activity,
    run_agent_graph,
    run_graph_until_interrupt as sourcing_run_graph_until_interrupt,
    resume_graph as sourcing_resume_graph,
    run_research_activity,
    score_suppliers_activity,
)
from app.agents.sourcing_agent.verification import (
    verify_po_exists,
    verify_payment_initiated,
)
from app.agents.sourcing_agent import TASK_QUEUE as SOURCING_TASK_QUEUE

# Hiring agent — same engine (Temporal + LangGraph + interrupt()), a harder
# problem: screen -> score -> schedule -> interview -> human review -> offer
from app.agents.hiring_agent.workflow import HiringWorkflow
from app.agents.hiring_agent.activity import (
    run_graph_until_interrupt as hiring_run_graph_until_interrupt,
    resume_graph as hiring_resume_graph,
    send_offer_activity,
)
from app.agents.hiring_agent import TASK_QUEUE as HIRING_TASK_QUEUE
from app.agents.hiring_agent.db import init_tables as init_hiring_tables

# Benchmark agent — minimal workflow for load testing
from app.agents.benchmark_agent.workflow import BenchmarkWorkflow, benchmark_activity
from app.agents.benchmark_agent import TASK_QUEUE as BENCHMARK_TASK_QUEUE

# Ensure DB tables are initialized when the worker starts
from app.agents.sourcing_agent.db import init_tables


async def main():
    await init_tables()
    await init_hiring_tables()
    client = await get_temporal_client()

    # Run all workers concurrently — sourcing, hiring, and benchmark each on
    # their own task queue
    import asyncio
    await asyncio.gather(
        run_worker(
            client=client,
            task_queue=SOURCING_TASK_QUEUE,
            workflows=[SourcingWorkflow],
            activities=[
                run_agent_graph,
                sourcing_run_graph_until_interrupt,
                sourcing_resume_graph,
                run_research_activity,
                create_po_activity,
                initiate_payment_activity,
                score_suppliers_activity,
                verify_po_exists,
                verify_payment_initiated,
            ],
        ),
        run_worker(
            client=client,
            task_queue=HIRING_TASK_QUEUE,
            workflows=[HiringWorkflow],
            activities=[
                hiring_run_graph_until_interrupt,
                hiring_resume_graph,
                send_offer_activity,
            ],
        ),
        run_worker(
            client=client,
            task_queue=BENCHMARK_TASK_QUEUE,
            workflows=[BenchmarkWorkflow],
            activities=[benchmark_activity],
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
