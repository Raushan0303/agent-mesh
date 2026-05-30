"""
Hello-world Temporal Workflow — smoke test.

This is a THROWAWAY script to verify the local Temporal cluster is running.
It is NOT part of the platform (app/agentmesh/) or the agent (app/agents/).
It lives in scripts/ and gets deleted after Phase 0 verification.

Usage:
    1. Start the cluster:  docker-compose up -d
    2. Run this script:    python scripts/hello_workflow.py
    3. Check the UI:       http://localhost:8080
"""

import asyncio
from datetime import timedelta

from temporalio import workflow, activity
from temporalio.client import Client
from temporalio.worker import Worker


@activity.defn
async def say_hello(name: str) -> str:
    """Trivial activity — just returns a greeting."""
    return f"Hello, {name}! AgentMesh Temporal cluster is working."


@activity.defn
async def ask_how_are_you(name: str) -> str:
    """Second activity — asks how the person is doing."""
    return f"How are you, {name}?"


@workflow.defn
class HelloWorldWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        # Step 1: say hello
        greeting = await workflow.execute_activity(
            say_hello,
            name,
            start_to_close_timeout=timedelta(seconds=10),
        )
        # Step 2: ask how are you
        question = await workflow.execute_activity(
            ask_how_are_you,
            name,
            start_to_close_timeout=timedelta(seconds=10),
        )
        return f"{greeting} | {question}"


async def main():
    # Connect to local Temporal cluster
    client = await Client.connect("localhost:7233")

    # Start a worker in the background to execute the workflow
    worker = Worker(
        client=client,
        task_queue="hello-task-queue",
        workflows=[HelloWorldWorkflow],
        activities=[say_hello, ask_how_are_you],
    )
    worker_task = asyncio.create_task(worker.run())

    # Give the worker a moment to start polling
    await asyncio.sleep(1)

    # Start the workflow (this blocks until completion)
    result = await client.execute_workflow(
        HelloWorldWorkflow.run,
        "Founder",
        id="hello-workflow-test-2",
        task_queue="hello-task-queue",
    )

    print(f"✅ Workflow completed: {result}")
    print(f"   View in UI: http://localhost:8080")
    print(f"   Workflow ID: hello-workflow-test")

    # Shutdown the worker
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
