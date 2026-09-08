"""HiringWorkflow — Temporal owns 100% of durable state (task_1 refactor).

Same human-in-the-loop shape as SourcingWorkflow, applied to a harder
problem: screen -> score -> schedule -> interview -> human review ->
offer -> send.

Flow:
  1. Activity: run_hiring_graph(initial_state)
     -> runs Screen Resume -> Score -> (Reject | Schedule -> Interview)
     -> hits the review gate, returns the FULL graph state (recorded
        in Temporal Event History)

  2. while the graph is paused:
     -> workflow.wait_condition() for the "approve" Signal (Temporal
        owns the wait — no checkpointer, no LangGraph persistence)
     -> Activity: run_hiring_graph(saved_state + approval)
        -> the graph's START router goes straight to Process Review
     -> on rejection the graph may loop back to Screen Resume and pause
        again — the while loop handles re-pauses natively

  3. If approved -> send_offer_activity
     If rejected -> return with status="rejected"
"""

import asyncio
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from app.agents.hiring_agent.activity import (
        run_hiring_graph,
        send_offer_activity,
    )
    from app.agents.hiring_agent.verification import verify_offer_sent
    from app.agents.hiring_agent.state import (
        HiringBriefInput,
        HiringResult,
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
class HiringWorkflow:
    """Hiring workflow with a human recruiter checkpoint.

    Signals:
      - "approve": sent by the agent-agnostic /approve endpoint with
        {"approved": bool, "comment": str} — same signal name as
        SourcingWorkflow, so the gateway route doesn't need to know
        which agent it's talking to.
    """

    @workflow.run
    async def run(self, brief: HiringBriefInput) -> HiringResult:
        graph_result = await workflow.execute_activity(
            run_hiring_graph,
            {
                "brief": brief.model_dump(),
                "screening_feedback": "",
                "followup_count": 0,
                "interview_score": 50.0,
                "interview_transcript": [],
                "rejection_count": 0,
            },
            start_to_close_timeout=GRAPH_START_TO_CLOSE,
            schedule_to_start_timeout=GRAPH_SCHEDULE_TO_START,
            schedule_to_close_timeout=GRAPH_SCHEDULE_TO_CLOSE,
            retry_policy=AGGRESSIVE_RETRY_TEMPLATE,
        )

        state = graph_result.get("state", {})

        # Pause loop — Temporal owns the wait. Each re-pause (e.g. after
        # a rejection-retry) waits for a fresh approval signal.
        while graph_result.get("paused", False):
            review_request = graph_result.get("review_request", {})
            workflow.logger.info(
                "WORKFLOW_PAUSED_AT_HUMAN_REVIEW waiting_for_signal candidate=%s",
                review_request.get("candidate_name", ""),
            )

            try:
                await workflow.wait_condition(
                    lambda: self._approval_received,
                    timeout=timedelta(hours=24),
                )
            except asyncio.TimeoutError:
                pass

            if not self._approval_received:
                return HiringResult(
                    candidate_name=brief.candidate_name,
                    role=brief.role,
                    status="timeout",
                    screening_score=graph_result.get("screening_score"),
                    interview_score=graph_result.get("interview_score"),
                    skills_matched=graph_result.get("skills_matched"),
                    approval_status="timeout",
                )

            # Inject the recruiter's decision into the saved state and
            # resume — a FRESH stateless invocation, not a checkpoint reload.
            self._approval_received = False
            resume_state = {**state, "approval": self._approval_data}
            graph_result = await workflow.execute_activity(
                run_hiring_graph,
                resume_state,
                start_to_close_timeout=GRAPH_START_TO_CLOSE,
                schedule_to_start_timeout=GRAPH_SCHEDULE_TO_START,
                schedule_to_close_timeout=GRAPH_SCHEDULE_TO_CLOSE,
                retry_policy=AGGRESSIVE_RETRY_TEMPLATE,
            )
            state = graph_result.get("state", {})

        final_status = graph_result.get("status", "failed")
        approval_status = graph_result.get("approval_status", "")

        if final_status != "awaiting_offer":
            # rejected_by_score, rejected (max retries), or failed
            return HiringResult(
                candidate_name=brief.candidate_name,
                role=brief.role,
                status=final_status,
                screening_score=graph_result.get("screening_score"),
                interview_score=graph_result.get("interview_score"),
                skills_matched=graph_result.get("skills_matched"),
                approval_status=approval_status or None,
                rejection_count=graph_result.get("rejection_count", 0),
            )

        # Approved — send the offer (side-effecting Activity, idempotent).
        # Falls back to the candidate's target salary if the graph's own
        # offer-decision amount is unavailable — the offer still goes out
        # instead of failing the workflow over a missing number.
        offer_amount = graph_result.get("offer_amount") or brief.target_salary
        offer_result = await workflow.execute_activity(
            send_offer_activity,
            args=(brief.candidate_name, brief.role, offer_amount),
            start_to_close_timeout=SIDE_EFFECT_START_TO_CLOSE,
            schedule_to_start_timeout=SIDE_EFFECT_SCHEDULE_TO_START,
            schedule_to_close_timeout=SIDE_EFFECT_SCHEDULE_TO_CLOSE,
            retry_policy=STRICT_NON_RETRYABLE_TEMPLATE,
        )

        # Verify the offer was actually sent (never trust the activity's self-report)
        offer_verification = await workflow.execute_activity(
            verify_offer_sent,
            args=(offer_result["offer_id"], brief.candidate_name, brief.role, offer_amount),
            start_to_close_timeout=READ_ONLY_START_TO_CLOSE,
            schedule_to_start_timeout=READ_ONLY_SCHEDULE_TO_START,
            schedule_to_close_timeout=READ_ONLY_SCHEDULE_TO_CLOSE,
            retry_policy=AGGRESSIVE_RETRY_TEMPLATE,
        )

        if not offer_verification.get("verified", False):
            ver_status = offer_verification.get("status", "failed")
            workflow.logger.error(
                "OFFER_VERIFICATION_%s offer_id=%s mismatches=%s",
                ver_status.upper(),
                offer_result["offer_id"],
                offer_verification.get("mismatches", []),
            )
            # UNKNOWN = we can't tell if the offer was sent — do NOT retry
            # (retrying could send a duplicate offer). Escalate to a human.
            return HiringResult(
                candidate_name=brief.candidate_name,
                role=brief.role,
                status=f"verification_{ver_status}",
                screening_score=graph_result.get("screening_score"),
                interview_score=graph_result.get("interview_score"),
                skills_matched=graph_result.get("skills_matched"),
                offer_id=offer_result["offer_id"],
                offer_amount=offer_amount,
                approval_status="approved",
            )

        # Evaluate the task contract — done_when conditions must pass
        # before status="completed". Empty contract = no checks (backward compatible).
        contract = TaskContract(
            done_when=[DoneCondition(check=c) for c in brief.done_when],
            escalate_when=[EscalateCondition(trigger=t) for t in brief.escalate_when],
        )
        contract_state = {
            "suppliers": [],
            "status": graph_result.get("status", ""),
            "rejection_count": graph_result.get("rejection_count", 0),
        }
        contract_evidence = {"offer_sent": offer_verification}
        contract_result = evaluate_contract(
            contract, contract_state, contract_evidence,
        )
        if contract_result.escalated:
            workflow.logger.warning(
                "CONTRACT_ESCALATED reason=%s", contract_result.escalation_reason,
            )
            return HiringResult(
                candidate_name=brief.candidate_name,
                role=brief.role,
                status="escalated",
                screening_score=graph_result.get("screening_score"),
                interview_score=graph_result.get("interview_score"),
                skills_matched=graph_result.get("skills_matched"),
                offer_id=offer_result["offer_id"],
                offer_amount=offer_amount,
                approval_status="approved",
            )
        if not contract_result.all_passed:
            workflow.logger.warning(
                "CONTRACT_UNMET failed=%s", contract_result.failed,
            )
            return HiringResult(
                candidate_name=brief.candidate_name,
                role=brief.role,
                status="contract_unmet",
                screening_score=graph_result.get("screening_score"),
                interview_score=graph_result.get("interview_score"),
                skills_matched=graph_result.get("skills_matched"),
                offer_id=offer_result["offer_id"],
                offer_amount=offer_amount,
                approval_status="approved",
            )

        return HiringResult(
            candidate_name=brief.candidate_name,
            role=brief.role,
            status="completed",
            screening_score=graph_result.get("screening_score"),
            interview_score=graph_result.get("interview_score"),
            skills_matched=graph_result.get("skills_matched"),
            offer_id=offer_result["offer_id"],
            offer_amount=offer_amount,
            approval_status="approved",
        )

    # ── Signal handler ──

    _approval_received: bool = False
    _approval_data: dict = {}

    @workflow.signal
    def approve(self, approval_data: dict) -> None:
        """Signal handler: receive the recruiter's decision.

        Called when the recruiter hits POST /workflows/{id}/approve with:
          {"approved": true, "comment": "Strong interview, extend offer"}
        """
        self._approval_data = approval_data
        self._approval_received = True
        workflow.logger.info(
            "APPROVAL_SIGNAL_RECEIVED approved=%s comment=%s",
            approval_data.get("approved", False),
            approval_data.get("comment", ""),
        )
