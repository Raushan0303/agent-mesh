"""HiringWorkflow — same human-in-the-loop shape as SourcingWorkflow, applied
to a harder problem: screen -> score -> schedule -> interview -> human
review -> offer -> send.

Flow:
  1. Activity: run_graph_until_interrupt
     -> runs Screen Resume -> Score -> (Reject | Schedule -> Interview)
     -> interrupt() fires at Human Review, graph pauses
     -> Activity returns paused=True + review_request

  2. Workflow waits for the "approve" Signal (recruiter calls the
     agent-agnostic /workflows/{id}/approve endpoint)

  3. Activity: resume_graph
     -> Command(resume=review_data) unblocks interrupt()
     -> runs Human Review (resumed) -> Offer Decision -> END

  4. If approved -> send_offer_activity
     If rejected -> return with status="rejected"
"""

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from app.agents.hiring_agent.activity import (
        resume_graph,
        run_graph_until_interrupt,
        send_offer_activity,
    )
    from app.agents.hiring_agent.state import (
        HiringBriefInput,
        HiringResult,
    )
    from app.core.constants import (
        AGGRESSIVE_RETRY_TEMPLATE,
        GRAPH_SCHEDULE_TO_CLOSE,
        GRAPH_SCHEDULE_TO_START,
        GRAPH_START_TO_CLOSE,
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
            run_graph_until_interrupt,
            brief,
            start_to_close_timeout=GRAPH_START_TO_CLOSE,
            schedule_to_start_timeout=GRAPH_SCHEDULE_TO_START,
            schedule_to_close_timeout=GRAPH_SCHEDULE_TO_CLOSE,
            retry_policy=AGGRESSIVE_RETRY_TEMPLATE,
        )

        if not graph_result.get("paused", False):
            return HiringResult(
                candidate_name=brief.candidate_name,
                role=brief.role,
                status=graph_result.get("status", "failed"),
                screening_score=graph_result.get("screening_score"),
                interview_score=graph_result.get("interview_score"),
                skills_matched=graph_result.get("skills_matched"),
            )

        review_request = graph_result.get("review_request", {})

        workflow.logger.info(
            "WORKFLOW_PAUSED_AT_HUMAN_REVIEW waiting_for_signal candidate=%s",
            review_request.get("candidate_name", ""),
        )

        await workflow.wait_condition(
            lambda: self._approval_received,
            timeout=timedelta(hours=24),
        )

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

        resume_result = await workflow.execute_activity(
            resume_graph,
            self._approval_data,
            start_to_close_timeout=GRAPH_START_TO_CLOSE,
            schedule_to_start_timeout=GRAPH_SCHEDULE_TO_START,
            schedule_to_close_timeout=GRAPH_SCHEDULE_TO_CLOSE,
            retry_policy=AGGRESSIVE_RETRY_TEMPLATE,
        )

        final_status = resume_result.get("status", "awaiting_offer")
        approval_status = resume_result.get("approval_status", "approved")

        if approval_status == "rejected" or final_status == "rejected":
            return HiringResult(
                candidate_name=brief.candidate_name,
                role=brief.role,
                status="rejected",
                screening_score=graph_result.get("screening_score"),
                interview_score=resume_result.get("interview_score", graph_result.get("interview_score")),
                skills_matched=graph_result.get("skills_matched"),
                approval_status="rejected",
                rejection_count=self._approval_data.get("rejection_count", 0),
            )

        # Approved — send the offer (side-effecting Activity, idempotent).
        # Falls back to the candidate's target salary if the graph's own
        # offer-decision amount is unavailable for any reason (e.g. a
        # checkpointer replay edge case) — the offer still goes out instead
        # of failing the workflow over a missing number.
        offer_amount = resume_result.get("offer_amount") or brief.target_salary
        offer_result = await workflow.execute_activity(
            send_offer_activity,
            args=(brief.candidate_name, brief.role, offer_amount),
            start_to_close_timeout=SIDE_EFFECT_START_TO_CLOSE,
            schedule_to_start_timeout=SIDE_EFFECT_SCHEDULE_TO_START,
            schedule_to_close_timeout=SIDE_EFFECT_SCHEDULE_TO_CLOSE,
            retry_policy=STRICT_NON_RETRYABLE_TEMPLATE,
        )

        return HiringResult(
            candidate_name=brief.candidate_name,
            role=brief.role,
            status="completed",
            screening_score=graph_result.get("screening_score"),
            interview_score=resume_result.get("interview_score", graph_result.get("interview_score")),
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
