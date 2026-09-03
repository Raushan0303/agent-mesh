from typing import TypedDict

from pydantic import BaseModel


class SourcingBriefInput(BaseModel):
    """Input schema — this is what the gateway validates against."""

    item: str
    quantity: int
    budget: float
    deadline: str | None = None
    cost_budget_usd: float = 0.0  # per-workflow cost budget; 0 = use default


class SourcingResult(BaseModel):
    """Output schema — returned by the Workflow."""

    suppliers: list[dict]
    status: str  # "completed" | "failed" | "no_matches" | "rejected" | "cost_exceeded" | "verification_failed"
    attempts: int
    po_id: str | None = None
    payment_id: str | None = None
    selected_supplier: str | None = None
    approval_status: str | None = None  # "approved" | "rejected" | None
    cost_incurred: float = 0.0
    cost_budget: float = 0.0
    verification_error: str = ""
    prompt_version: str = ""  # which prompt version was used for the decide node


class ResearchResult(BaseModel):
    """Output of the research Activity — passed to the PO Activity."""

    suppliers: list[dict]
    status: str
    attempts: int


class PurchaseOrderResult(BaseModel):
    """Output of the create_po Activity."""

    po_id: str
    supplier_name: str
    status: str


class PaymentResult(BaseModel):
    """Output of the initiate_payment Activity."""

    payment_id: str
    po_id: str
    status: str


class ScoredSuppliersResult(BaseModel):
    """Output of the score_suppliers Activity (versioned)."""

    suppliers: list[dict]


class AgentState(TypedDict, total=False):
    """LangGraph state — flows between nodes.

    Week 4: extended with score, decision, and approval fields.
    Each node reads what it needs and writes what it produces.
    """

    # Input
    brief: SourcingBriefInput
    trace_id: str  # W3C trace ID for cross-service tracing (InferRoute)

    # Research node output
    suppliers: list[dict]
    attempts: int
    status: str  # "running" | "completed" | "no_matches"

    # Score node output
    scored_suppliers: list[dict]

    # Decide node output
    selected_supplier: dict
    decision_reason: str
    prompt_hash: str  # SHA256[:16] of the rendered decide prompt (prompt versioning)
    prompt_version: str  # human-readable version tag (e.g. "v1")

    # Approve node output
    approval_status: str  # "approved" | "rejected"
    approval_comment: str
    rejection_count: int  # incremented on each rejection, used for retry limit

    # Confirm node output
    po_id: str
    payment_id: str
    final_status: str  # "completed" | "rejected"

    # Cost tracking (Week 12-13)
    cost_incurred: float  # accumulated LLM cost within this graph run
