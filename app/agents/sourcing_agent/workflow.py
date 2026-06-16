"""SourcingWorkflow — Week 4: human-in-the-loop with interrupt() + Signals.

Flow:
  1. Activity: run_graph_until_interrupt
     → runs Research → Score → Decide → Approve
     → interrupt() fires, graph pauses
     → Activity returns with paused=True + approval_request

  2. Workflow waits for "approve" Signal (human calls /approve endpoint)

  3. Activity: resume_graph
     → Command(resume=approval_data) unblocks interrupt()
     → runs Approve (resumed) → Confirm → END

  4. If approved → create_po_activity + initiate_payment_activity
     If rejected → return with status="rejected"
"""

import asyncio
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from app.agentmesh.temporal.versioning import is_patched
    from app.agents.sourcing_agent.activity import (
        create_po_activity,
        initiate_payment_activity,
        run_agent_graph,
        run_graph_until_interrupt,
        resume_graph,
        run_research_activity,
        score_suppliers_activity,
    )
    from app.agents.sourcing_agent.state import (
        SourcingBriefInput,
        SourcingResult,
    )
    from app.core.constants import (
        AGGRESSIVE_RETRY_TEMPLATE,
        DEFAULT_ACTIVITY_TIMEOUT,
        GRAPH_SCHEDULE_TO_CLOSE,
        GRAPH_SCHEDULE_TO_START,
        GRAPH_START_TO_CLOSE,
        READ_ONLY_SCHEDULE_TO_CLOSE,
        READ_ONLY_SCHEDULE_TO_START,
        READ_ONLY_START_TO_CLOSE,
        SIDE_EFFECT_SCHEDULE_TO_CLOSE,
        SIDE_EFFECT_SCHEDULE_TO_START,
        SIDE_EFFECT_START_TO_CLOSE,
        STRICT_NON_RETRYABLE_TEMPLATE,
        STRICT_RETRY_TEMPLATE,
    )


@workflow.defn
class SourcingWorkflow:
    """Sourcing workflow with human-in-the-loop approval.

    Signals:
      - "approve": sent by the /approve endpoint with {"approved": bool, "comment": str}
    """

    @workflow.run
    async def run(self, brief: SourcingBriefInput) -> SourcingResult:
        """Run the 5-node graph with human approval checkpoint."""
        # Activity 1: Run the graph until interrupt() or completion
        # Graph activities get longer timeouts (the graph runs multiple tools)
        graph_result = await workflow.execute_activity(
            run_graph_until_interrupt,
            brief,
            start_to_close_timeout=GRAPH_START_TO_CLOSE,
            schedule_to_start_timeout=GRAPH_SCHEDULE_TO_START,
            schedule_to_close_timeout=GRAPH_SCHEDULE_TO_CLOSE,
            retry_policy=AGGRESSIVE_RETRY_TEMPLATE,
        )

        # If the graph completed without pausing (no_matches), return early
        if not graph_result.get("paused", False):
            return SourcingResult(
                suppliers=graph_result.get("suppliers", []),
                status=graph_result.get("status", "failed"),
                attempts=graph_result.get("attempts", 0),
            )

        # The graph paused at Approve — wait for human signal
        approval_request = graph_result.get("approval_request", {})
        selected = graph_result.get("selected_supplier", {})

        workflow.logger.info(
            "WORKFLOW_PAUSED_AT_APPROVE waiting_for_signal supplier=%s",
            approval_request.get("supplier", ""),
        )

        # Wait for the "approve" signal
        approval_data = await workflow.wait_condition(
            lambda: self._approval_received,
            timeout=timedelta(hours=24),
        )

        if not self._approval_received:
            # Timeout — no approval received in 24 hours
            return SourcingResult(
                suppliers=graph_result.get("suppliers", []),
                status="timeout",
                attempts=graph_result.get("attempts", 0),
                selected_supplier=selected.get("name") if selected else None,
                approval_status="timeout",
            )

        # Activity 2: Resume the graph with the approval data
        resume_result = await workflow.execute_activity(
            resume_graph,
            self._approval_data,
            start_to_close_timeout=GRAPH_START_TO_CLOSE,
            schedule_to_start_timeout=GRAPH_SCHEDULE_TO_START,
            schedule_to_close_timeout=GRAPH_SCHEDULE_TO_CLOSE,
            retry_policy=AGGRESSIVE_RETRY_TEMPLATE,
        )

        final_status = resume_result.get("status", "completed")
        approval_status = resume_result.get("approval_status", "approved")

        # If rejected, return without creating PO or payment
        if final_status == "rejected" or approval_status == "rejected":
            return SourcingResult(
                suppliers=graph_result.get("suppliers", []),
                status="rejected",
                attempts=graph_result.get("attempts", 0),
                selected_supplier=selected.get("name") if selected else None,
                approval_status="rejected",
            )

        # Approved — create PO and initiate payment
        # Side-effecting tools get careful timeouts (longer StartToClose,
        # careful ScheduleToClose to allow limited retries)
        unit_price = selected.get("price", 0)
        po_result = await workflow.execute_activity(
            create_po_activity,
            args=(selected.get("name", ""), brief.item, brief.quantity, unit_price),
            start_to_close_timeout=SIDE_EFFECT_START_TO_CLOSE,
            schedule_to_start_timeout=SIDE_EFFECT_SCHEDULE_TO_START,
            schedule_to_close_timeout=SIDE_EFFECT_SCHEDULE_TO_CLOSE,
            retry_policy=STRICT_NON_RETRYABLE_TEMPLATE,
        )

        total_amount = unit_price * brief.quantity
        payment_result = await workflow.execute_activity(
            initiate_payment_activity,
            args=(po_result["po_id"], total_amount),
            start_to_close_timeout=SIDE_EFFECT_START_TO_CLOSE,
            schedule_to_start_timeout=SIDE_EFFECT_SCHEDULE_TO_START,
            schedule_to_close_timeout=SIDE_EFFECT_SCHEDULE_TO_CLOSE,
            retry_policy=STRICT_NON_RETRYABLE_TEMPLATE,
        )

        return SourcingResult(
            suppliers=graph_result.get("suppliers", []),
            status="completed",
            attempts=graph_result.get("attempts", 0),
            po_id=po_result["po_id"],
            payment_id=payment_result["payment_id"],
            selected_supplier=selected.get("name") if selected else None,
            approval_status="approved",
        )

    # ── Signal handler ──

    _approval_received: bool = False
    _approval_data: dict = {}

    @workflow.signal
    def approve(self, approval_data: dict) -> None:
        """Signal handler: receive human approval decision.

        Called when the human hits POST /workflows/{id}/approve with:
          {"approved": true, "comment": "looks good"}
        """
        self._approval_data = approval_data
        self._approval_received = True
        workflow.logger.info(
            "APPROVAL_SIGNAL_RECEIVED approved=%s comment=%s",
            approval_data.get("approved", False),
            approval_data.get("comment", ""),
        )
