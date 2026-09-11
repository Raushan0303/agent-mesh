from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel


class RiskTier(str, Enum):
    """Risk classification for tools — determines the authorization path.

    READ_ONLY         — query tools (query_suppliers, get_price_quote)
    WRITE_REVERSIBLE  — update preferences, reversible writes
    SIDE_EFFECT       — create_purchase_order (creates external state)
    IRREVERSIBLE      — initiate_payment, send_offer (money moves, can't undo)
    """
    READ_ONLY = "read_only"
    WRITE_REVERSIBLE = "write_reversible"
    SIDE_EFFECT = "side_effect"
    IRREVERSIBLE = "irreversible"


class AuthorizationMode(str, Enum):
    """How the harness authorizes a tool call.

    AUTOMATIC           — no approval needed (read-only, reversible)
    APPROVAL_REQUIRED   — must pause for human approval before executing
    """
    AUTOMATIC = "automatic"
    APPROVAL_REQUIRED = "approval_required"


@dataclass
class ToolSpec:
    """Specification for a tool registered in the Tool Registry.

    The registry uses this to validate inputs, enforce timeouts,
    and validate outputs. The spec is agent-agnostic — it contains
    no domain logic, only structural metadata.

    Risk tiers (task_7 phase 3): every tool is classified by risk.
    Read-only tools auto-run. Irreversible tools can require human
    approval before executing — the permission ladder from the
    harness engineering framework.
    """

    name: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    timeout_seconds: float
    idempotency_required: bool
    allowed_egress: list[str] = field(default_factory=list)
    max_output_bytes: int = 65536
    risk_tier: RiskTier = RiskTier.READ_ONLY
    authorization_mode: AuthorizationMode = AuthorizationMode.AUTOMATIC
