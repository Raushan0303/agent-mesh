from pydantic import BaseModel, Field


# ── schedule_interview ──


class ScheduleInterviewInput(BaseModel):
    candidate_name: str = Field(..., description="Full name of the candidate")
    role: str = Field(..., description="Role being interviewed for")


class ScheduleInterviewOutput(BaseModel):
    candidate_name: str
    role: str
    slot: str
    interviewer: str
    calendar_id: str


# ── send_offer ──


class SendOfferInput(BaseModel):
    candidate_name: str
    role: str
    offer_amount: float = Field(..., gt=0)


class SendOfferOutput(BaseModel):
    offer_id: str
    candidate_name: str
    status: str  # "sent" | "failed"
