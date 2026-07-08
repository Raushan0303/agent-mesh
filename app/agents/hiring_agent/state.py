from typing import TypedDict

from pydantic import BaseModel


class HiringBriefInput(BaseModel):
    """Input schema — this is what the gateway validates against."""

    candidate_name: str
    role: str
    resume_text: str
    years_experience: float
    target_salary: float
    notice_period_days: int | None = None


class HiringResult(BaseModel):
    """Output schema — returned by the Workflow."""

    candidate_name: str
    role: str
    status: str  # "completed" | "rejected_by_score" | "rejected" | "timeout" | "failed"
    screening_score: float | None = None
    interview_score: float | None = None
    skills_matched: list[str] | None = None
    offer_id: str | None = None
    offer_amount: float | None = None
    approval_status: str | None = None  # "approved" | "rejected" | None
    rejection_count: int = 0


class AgentState(TypedDict, total=False):
    """LangGraph state — flows between nodes.

    Same shape philosophy as the sourcing agent: each node reads what it
    needs and writes what it produces. The hiring agent applies the exact
    same engine (Temporal durability + LangGraph nodes + interrupt() for
    human checkpoints) to a harder, higher-stakes workflow.
    """

    # Input
    brief: HiringBriefInput
    trace_id: str

    # Screen resume node output (agentic — LLM extracts signal from resume)
    screening_notes: str
    skills_matched: list[str]
    screening_feedback: str  # feedback from a previous human rejection, if any

    # Score rubric node output (deterministic — NO LLM)
    screening_score: float

    # Schedule interview node output (tool call via registry)
    interview_slot: dict

    # Interview node output (agentic, multi-turn with a bounded follow-up loop)
    interview_transcript: list[dict]
    followup_count: int
    interview_score: float
    interview_done: bool

    # Human review node output (interrupt() checkpoint)
    approval_status: str  # "approved" | "rejected"
    approval_comment: str
    rejection_count: int  # incremented on each rejection, used for retry limit

    # Offer decision node output (agentic)
    offer_amount: float
    decision_reason: str

    # Terminal status set by reject / offer_decision nodes
    final_status: str  # "rejected_by_score" | "awaiting_offer" | "rejected"
