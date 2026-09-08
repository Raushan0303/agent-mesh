"""SourcingWorkflow — Temporal owns 100% of durable state (task_1 refactor).

Flow:
  1. Activity: run_sourcing_graph(initial_state)
     → runs Research → Score → Decide → hits the approval gate
     → returns the FULL graph state (recorded in Temporal Event History)

  2. while the graph is paused:
     → workflow.wait_condition() for the "approve" Signal (Temporal owns
       the wait — no checkpointer, no LangGraph persistence)
     → Activity: run_sourcing_graph(saved_state + approval)
       → the graph's START router goes straight to Process Approval
     → on rejection the graph may loop back to Research and pause again —
       the while loop handles re-pauses natively

  3. If approved → create_po_activity + initiate_payment_activity
     If rejected → return with status="rejected"

The LangGraph graph is a pure in-memory reasoning unit inside the
Activity. State crosses the Activity boundary as a plain dict — Temporal
records it, Temporal owns it.
"""

import asyncio
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from app.agents.sourcing_agent.activity import (
        create_po_activity,
        initiate_payment_activity,
        run_agent_graph,
        run_research_activity,
        run_sourcing_graph,
        score_suppliers_activity,
    )
    from app.agents.sourcing_agent.verification import (
        verify_po_exists,
        verify_payment_initiated,
    )
    from app.agents.sourcing_agent.state import (
        SourcingBriefInput,
        SourcingResult,
    )
    from app.agentmesh.contract import TaskContract, DoneCondition, EscalateCondition, evaluate_contract
    from app.core.constants import (
        AGGRESSIVE_RETRY_TEMPLATE,
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
    )


@workflow.defn
class SourcingWorkflow:
    """Sourcing workflow with human-in-the-loop approval.

    Signals:
      - "approve": sent by the /approve endpoint with {"approved": bool, "comment": str}
    """

    @workflow.run
    async def run(self, brief: SourcingBriefInput) -> SourcingResult:
        """Run the graph with Temporal-owned pauses for human approval."""
        # Resolve cost budget (0 = use default from settings)
        cost_budget = brief.cost_budget_usd
        if cost_budget <= 0:
            from app.core.config import settings
            cost_budget = settings.default_cost_budget_usd

        total_cost = 0.0

        # Activity: run the graph until the approval gate (or completion).
        # The state crosses the boundary as a plain dict — Temporal
        # records the full state in Event History.
        graph_result = await workflow.execute_activity(
            run_sourcing_graph,
            {
                "brief": brief.model_dump(),
                "trace_id": getattr(brief, "trace_id", "") or "",
                "suppliers": [],
                "attempts": 0,
                "status": "running",
            },
            start_to_close_timeout=GRAPH_START_TO_CLOSE,
            schedule_to_start_timeout=GRAPH_SCHEDULE_TO_START,
            schedule_to_close_timeout=GRAPH_SCHEDULE_TO_CLOSE,
            retry_policy=AGGRESSIVE_RETRY_TEMPLATE,
        )

        total_cost += graph_result.get("cost_incurred", 0.0)
        state = graph_result.get("state", {})

        # Pause loop — Temporal owns the wait. Each re-pause (e.g. after
        # a rejection-retry) waits for a fresh approval signal.
        while graph_result.get("paused", False):
            # Cost check before waiting on a human
            if total_cost > cost_budget:
                workflow.logger.warning(
                    "COST_BUDGET_EXCEEDED spent=%.4f budget=%.4f (paused at approval gate)",
                    total_cost, cost_budget,
                )
                return SourcingResult(
                    suppliers=graph_result.get("suppliers", []),
                    status="cost_exceeded",
                    attempts=graph_result.get("attempts", 0),
                    cost_incurred=total_cost,
                    cost_budget=cost_budget,
                )

            approval_request = graph_result.get("approval_request", {})
            workflow.logger.info(
                "WORKFLOW_PAUSED_AT_APPROVAL waiting_for_signal supplier=%s",
                approval_request.get("supplier", ""),
            )

            # Wait for the "approve" signal — Temporal owns the wait.
            try:
                await workflow.wait_condition(
                    lambda: self._approval_received,
                    timeout=timedelta(hours=24),
                )
            except asyncio.TimeoutError:
                pass

            if not self._approval_received:
                # Timeout — no approval received in 24 hours
                selected = state.get("selected_supplier", {})
                return SourcingResult(
                    suppliers=graph_result.get("suppliers", []),
                    status="timeout",
                    attempts=graph_result.get("attempts", 0),
                    selected_supplier=selected.get("name") if selected else None,
                    approval_status="timeout",
                    cost_incurred=total_cost,
                    cost_budget=cost_budget,
                )

            # Inject the human's decision into the saved state and resume —
            # a FRESH stateless invocation, not a checkpoint reload.
            self._approval_received = False
            resume_state = {**state, "approval": self._approval_data}
            graph_result = await workflow.execute_activity(
                run_sourcing_graph,
                resume_state,
                start_to_close_timeout=GRAPH_START_TO_CLOSE,
                schedule_to_start_timeout=GRAPH_SCHEDULE_TO_START,
                schedule_to_close_timeout=GRAPH_SCHEDULE_TO_CLOSE,
                retry_policy=AGGRESSIVE_RETRY_TEMPLATE,
            )

            total_cost += graph_result.get("cost_incurred", 0.0)
            state = graph_result.get("state", {})

        # The graph finished — extract terminal state
        final_status = graph_result.get("status", "failed")
        approval_status = graph_result.get("approval_status", "")
        selected = graph_result.get("selected_supplier") or state.get("selected_supplier", {})

        # Cost check after graph completion
        if total_cost > cost_budget:
            workflow.logger.warning(
                "COST_BUDGET_EXCEEDED spent=%.4f budget=%.4f (after graph)",
                total_cost, cost_budget,
            )
            return SourcingResult(
                suppliers=graph_result.get("suppliers", []),
                status="cost_exceeded",
                attempts=graph_result.get("attempts", 0),
                selected_supplier=selected.get("name") if selected else None,
                cost_incurred=total_cost,
                cost_budget=cost_budget,
            )

        # If the graph ended without approval (no_matches, or rejected
        # after max retries), return without side effects
        if final_status != "completed":
            return SourcingResult(
                suppliers=graph_result.get("suppliers", []),
                status=final_status,
                attempts=graph_result.get("attempts", 0),
                selected_supplier=selected.get("name") if selected else None,
                approval_status=approval_status or None,
                cost_incurred=total_cost,
                cost_budget=cost_budget,
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

        # Verify the PO was actually created (never trust the activity's self-report)
        # Now with field-level checks — not just existence, but correctness
        po_verification = await workflow.execute_activity(
            verify_po_exists,
            args=(po_result["po_id"], selected.get("name", ""), brief.item, brief.quantity, unit_price),
            start_to_close_timeout=READ_ONLY_START_TO_CLOSE,
            schedule_to_start_timeout=READ_ONLY_SCHEDULE_TO_START,
            schedule_to_close_timeout=READ_ONLY_SCHEDULE_TO_CLOSE,
            retry_policy=AGGRESSIVE_RETRY_TEMPLATE,
        )

        if not po_verification.get("verified", False):
            po_status = po_verification.get("status", "failed")
            workflow.logger.error(
                "PO_VERIFICATION_%s po_id=%s mismatches=%s",
                po_status.upper(),
                po_result["po_id"],
                po_verification.get("mismatches", []),
            )
            # UNKNOWN = we can't tell if the PO was created — do NOT retry
            # (retrying could create a duplicate). Escalate to a human.
            return SourcingResult(
                suppliers=graph_result.get("suppliers", []),
                status=f"verification_{po_status}",
                attempts=graph_result.get("attempts", 0),
                selected_supplier=selected.get("name") if selected else None,
                approval_status="approved",
                verification_error=f"PO {po_result['po_id']} verification {po_status}: {po_verification.get('mismatches', [])}",
                cost_incurred=total_cost,
                cost_budget=cost_budget,
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

        # Verify the payment was actually initiated
        # Now with amount check — catches partial writes and stale results
        payment_verification = await workflow.execute_activity(
            verify_payment_initiated,
            args=(payment_result["payment_id"], total_amount),
            start_to_close_timeout=READ_ONLY_START_TO_CLOSE,
            schedule_to_start_timeout=READ_ONLY_SCHEDULE_TO_START,
            schedule_to_close_timeout=READ_ONLY_SCHEDULE_TO_CLOSE,
            retry_policy=AGGRESSIVE_RETRY_TEMPLATE,
        )

        if not payment_verification.get("verified", False):
            pay_status = payment_verification.get("status", "failed")
            workflow.logger.error(
                "PAYMENT_VERIFICATION_%s payment_id=%s mismatches=%s",
                pay_status.upper(),
                payment_result["payment_id"],
                payment_verification.get("mismatches", []),
            )
            # UNKNOWN = we can't tell if the payment was initiated — do NOT
            # retry (retrying could double-charge). Escalate to a human.
            return SourcingResult(
                suppliers=graph_result.get("suppliers", []),
                status=f"verification_{pay_status}",
                attempts=graph_result.get("attempts", 0),
                po_id=po_result["po_id"],
                selected_supplier=selected.get("name") if selected else None,
                approval_status="approved",
                verification_error=f"Payment {payment_result['payment_id']} verification {pay_status}: {payment_verification.get('mismatches', [])}",
                cost_incurred=total_cost,
                cost_budget=cost_budget,
            )

        # Evaluate the task contract — done_when conditions must pass
        # before status="completed". Empty contract = no checks (backward compatible).
        contract = TaskContract(
            done_when=[DoneCondition(check=c) for c in brief.done_when],
            escalate_when=[EscalateCondition(trigger=t) for t in brief.escalate_when],
        )
        contract_state = {
            "suppliers": graph_result.get("suppliers", []),
            "status": graph_result.get("status", ""),
            "rejection_count": graph_result.get("rejection_count", 0),
        }
        contract_evidence = {
            "po_created": po_verification,
            "payment_initiated": payment_verification,
        }
        contract_result = evaluate_contract(
            contract, contract_state, contract_evidence, total_cost, cost_budget,
        )
        if contract_result.escalated:
            workflow.logger.warning(
                "CONTRACT_ESCALATED reason=%s", contract_result.escalation_reason,
            )
            return SourcingResult(
                suppliers=graph_result.get("suppliers", []),
                status="escalated",
                attempts=graph_result.get("attempts", 0),
                po_id=po_result["po_id"],
                payment_id=payment_result["payment_id"],
                selected_supplier=selected.get("name") if selected else None,
                approval_status="approved",
                verification_error=contract_result.escalation_reason,
                cost_incurred=total_cost,
                cost_budget=cost_budget,
                prompt_version=graph_result.get("prompt_version", ""),
            )
        if not contract_result.all_passed:
            workflow.logger.warning(
                "CONTRACT_UNMET failed=%s", contract_result.failed,
            )
            return SourcingResult(
                suppliers=graph_result.get("suppliers", []),
                status="contract_unmet",
                attempts=graph_result.get("attempts", 0),
                po_id=po_result["po_id"],
                payment_id=payment_result["payment_id"],
                selected_supplier=selected.get("name") if selected else None,
                approval_status="approved",
                verification_error=f"Failed checks: {contract_result.failed}",
                cost_incurred=total_cost,
                cost_budget=cost_budget,
                prompt_version=graph_result.get("prompt_version", ""),
            )

        return SourcingResult(
            suppliers=graph_result.get("suppliers", []),
            status="completed",
            attempts=graph_result.get("attempts", 0),
            po_id=po_result["po_id"],
            payment_id=payment_result["payment_id"],
            selected_supplier=selected.get("name") if selected else None,
            approval_status="approved",
            cost_incurred=total_cost,
            cost_budget=cost_budget,
            prompt_version=graph_result.get("prompt_version", ""),
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
