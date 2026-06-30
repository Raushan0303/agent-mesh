from collections.abc import Callable

from temporalio.client import Client
from temporalio.worker import Worker

from app.core.config import settings


async def run_worker(
    client: Client,
    task_queue: str,
    workflows: list[type],
    activities: list[Callable],
    max_concurrent_activities: int | None = None,
    max_concurrent_workflow_tasks: int | None = None,
) -> None:
    """Start a Temporal Worker that polls the given task queue.

    This function is agent-agnostic — it accepts workflows and activities
    as parameters. The caller (worker_runner.py) imports the agent's
    Workflow and Activity classes and passes them in.

    Args:
        max_concurrent_activities: Override for max concurrent activities.
            Defaults to settings.max_concurrent_activities.
        max_concurrent_workflow_tasks: Override for max concurrent workflow
            tasks. Defaults to Temporal SDK default (100).
    """
    worker = Worker(
        client=client,
        task_queue=task_queue,
        workflows=workflows,
        activities=activities,
        max_concurrent_activities=max_concurrent_activities or settings.max_concurrent_activities,
        max_concurrent_workflow_tasks=max_concurrent_workflow_tasks or 100,
    )
    await worker.run()
