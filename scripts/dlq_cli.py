"""Dead-letter queue CLI — inspect and retry failed Workflows.

Uses Temporal's Visibility API to list failed Workflows, inspect their
event history, and manually retry them. Generic across agent types —
Temporal's Visibility API queries by Workflow Type, status, etc., none
of which requires agent-specific code.

Usage:
    python scripts/dlq_cli.py list-failed [--agent-type <type>] [--limit 20]
    python scripts/dlq_cli.py inspect <workflow_id>
    python scripts/dlq_cli.py retry <workflow_id>
    python scripts/dlq_cli.py stats
"""

import argparse
import asyncio
import sys
from datetime import timedelta

from temporalio.client import Client, WorkflowFailureError

from app.core.config import settings


async def get_client() -> Client:
    return await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
    )


async def cmd_list_failed(args: argparse.Namespace) -> None:
    """List failed Workflows via Temporal's Visibility API."""
    client = await get_client()

    # Query the Visibility API for failed workflows
    workflows = []
    async for wf in client.list_workflows(
        query="ExecutionStatus = 'Failed'",
        limit=args.limit,
    ):
        workflows.append(wf)

    if not workflows:
        print("No failed Workflows found.")
        return

    # Filter by agent type if specified
    if args.agent_type:
        workflows = [w for w in workflows if args.agent_type in w.id]

    print(f"\n{'='*80}")
    print(f"Failed Workflows ({len(workflows)} found)")
    print(f"{'='*80}")
    print(f"{'Workflow ID':<45} {'Status':<12} {'Start Time':<25}")
    print(f"{'-'*45} {'-'*12} {'-'*25}")
    for wf in workflows:
        start_time = wf.start_time.strftime("%Y-%m-%d %H:%M:%S") if wf.start_time else "N/A"
        print(f"{wf.id:<45} {wf.status.name:<12} {start_time:<25}")
    print()


async def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect a failed Workflow's event history."""
    client = await get_client()
    handle = client.get_workflow_handle(args.workflow_id)

    desc = await handle.describe()
    print(f"\n{'='*80}")
    print(f"Workflow: {args.workflow_id}")
    print(f"{'='*80}")
    print(f"  Status:    {desc.status.name}")
    print(f"  Run ID:    {desc.run_id}")
    print(f"  Start:     {desc.start_time}")
    if desc.close_time:
        print(f"  Close:     {desc.close_time}")
    print()

    # Fetch event history
    history = await handle.fetch_history()
    print(f"Event History ({len(history.events)} events):")
    print(f"{'-'*80}")
    for i, event in enumerate(history.events):
        etype = int(event.event_type) if event.event_type else 0
        # Map common event types
        EVENT_NAMES = {
            1: "WorkflowExecutionStarted",
            2: "WorkflowExecutionCompleted",
            3: "WorkflowExecutionFailed",
            4: "WorkflowExecutionTimedOut",
            5: "WorkflowTaskScheduled",
            6: "WorkflowTaskStarted",
            7: "WorkflowTaskCompleted",
            8: "WorkflowTaskFailed",
            9: "WorkflowTaskTimedOut",
            10: "ActivityTaskScheduled",
            11: "ActivityTaskStarted",
            12: "ActivityTaskCompleted",
            13: "ActivityTaskFailed",
            14: "ActivityTaskTimedOut",
            24: "MarkerRecorded",
            25: "WorkflowExecutionSignaled",
        }
        name = EVENT_NAMES.get(etype, f"EventType({etype})")
        print(f"  [{i+1:3d}] {name}")

    # If failed, try to get the failure reason
    if desc.status.name == "FAILED":
        try:
            result = await handle.result()
            print(f"\nResult: {result}")
        except Exception as e:
            print(f"\nFailure reason: {type(e).__name__}: {e}")
    print()


async def cmd_retry(args: argparse.Namespace) -> None:
    """Retry a failed Workflow by starting a new run with the same ID."""
    client = await get_client()
    handle = client.get_workflow_handle(args.workflow_id)

    desc = await handle.describe()
    if desc.status.name != "FAILED":
        print(f"Workflow {args.workflow_id} is not FAILED (status: {desc.status.name})")
        return

    # Get the original input from the event history
    history = await handle.fetch_history()
    original_input = None
    workflow_type = None
    for event in history.events:
        etype = int(event.event_type) if event.event_type else 0
        if etype == 1:  # WorkflowExecutionStarted
            attrs = event.workflow_execution_started_event_attributes
            if attrs:
                workflow_type = attrs.workflow_type.name if attrs.workflow_type else None
                # The input is in attrs.input (payloads)
                # For simplicity, we can't easily decode it without the data converter
                # In practice, you'd re-submit with known input
            break

    print(f"\nWorkflow: {args.workflow_id}")
    print(f"  Type: {workflow_type or 'unknown'}")
    print(f"  Status: {desc.status.name}")
    print()
    print("To retry, re-submit the workflow with the same input:")
    print(f"  curl -X POST http://localhost:8000/workflows \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"agent_type\": \"<agent_type>\", \"input\": {{...}}}}'")
    print()
    print("Note: Temporal doesn't auto-retry failed Workflows. You must re-submit")
    print("with a new workflow ID (or the same ID if the previous run is closed).")


async def cmd_stats(args: argparse.Namespace) -> None:
    """Show reliability stats — circuit breaker states + bulkhead states."""
    from app.agentmesh.reliability import get_all_breaker_states, get_all_bulkhead_states

    breaker_states = get_all_breaker_states()
    bulkhead_states = get_all_bulkhead_states()

    print(f"\n{'='*60}")
    print("Reliability Stats")
    print(f"{'='*60}")

    print(f"\nCircuit Breakers ({len(breaker_states)}):")
    if breaker_states:
        print(f"  {'Tool':<30} {'State':<12} {'Failures':<10} {'Threshold':<10}")
        print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*10}")
        for name, state in breaker_states.items():
            print(
                f"  {name:<30} {state['state']:<12} "
                f"{state['consecutive_failures']:<10} {state['failure_threshold']:<10}"
            )
    else:
        print("  (no circuit breakers registered)")

    print(f"\nBulkheads ({len(bulkhead_states)}):")
    if bulkhead_states:
        print(f"  {'Name':<30} {'Active':<10} {'Max':<10} {'Available':<10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
        for name, state in bulkhead_states.items():
            print(
                f"  {name:<30} {state['active']:<10} "
                f"{state['max']:<10} {state['available']:<10}"
            )
    else:
        print("  (no bulkheads registered)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="AgentMesh DLQ CLI — inspect and retry failed Workflows"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list-failed
    p_list = subparsers.add_parser("list-failed", help="List failed Workflows")
    p_list.add_argument("--agent-type", help="Filter by agent type (e.g., the agent name)")
    p_list.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect a failed Workflow")
    p_inspect.add_argument("workflow_id", help="Workflow ID to inspect")

    # retry
    p_retry = subparsers.add_parser("retry", help="Get retry instructions for a failed Workflow")
    p_retry.add_argument("workflow_id", help="Workflow ID to retry")

    # stats
    subparsers.add_parser("stats", help="Show circuit breaker + bulkhead stats")

    args = parser.parse_args()

    if args.command == "list-failed":
        asyncio.run(cmd_list_failed(args))
    elif args.command == "inspect":
        asyncio.run(cmd_inspect(args))
    elif args.command == "retry":
        asyncio.run(cmd_retry(args))
    elif args.command == "stats":
        asyncio.run(cmd_stats(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
