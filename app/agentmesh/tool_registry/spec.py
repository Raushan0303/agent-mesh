from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class ToolSpec:
    """Specification for a tool registered in the Tool Registry.

    The registry uses this to validate inputs, enforce timeouts,
    and validate outputs. The spec is agent-agnostic — it contains
    no domain logic, only structural metadata.
    """

    name: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    timeout_seconds: float
    idempotency_required: bool
    allowed_egress: list[str] = field(default_factory=list)
    max_output_bytes: int = 65536
