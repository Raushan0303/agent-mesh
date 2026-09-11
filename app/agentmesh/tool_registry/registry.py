import logging
from collections.abc import Awaitable, Callable

from app.agentmesh.tool_registry.egress import EgressFilter, SandboxedHTTPClient
from app.agentmesh.tool_registry.exceptions import ToolApprovalRequiredError, ToolNotFoundError
from app.agentmesh.tool_registry.output_cap import enforce_output_cap
from app.agentmesh.tool_registry.sandbox import run_with_timeout
from app.agentmesh.tool_registry.schema_validation import validate_input, validate_output
from app.agentmesh.tool_registry.spec import AuthorizationMode, ToolSpec

logger = logging.getLogger("agentmesh.tool_registry")


class ToolRegistry:
    """Generic tool registry — the single harness layer with no off-the-shelf equivalent.

    Any agent's LangGraph nodes call through this registry to invoke tools.
    The registry handles:
    1. Input schema validation (before execution)
    2. Egress filtering (creates SandboxedHTTPClient per tool)
    3. Timeout enforcement (during execution)
    4. Output size capping (after execution, before validation)
    5. Output schema validation (after execution)

    The registry contains ZERO agent-specific code. Tools register themselves;
    the registry just enforces the contract.
    """

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, Callable[..., Awaitable]]] = {}
        self._approved_tools: set[str] = set()

    def approve_tool(self, name: str) -> None:
        """Pre-approve a tool for execution (called by the workflow before
        calling a tool with authorization_mode=APPROVAL_REQUIRED)."""
        self._approved_tools.add(name)

    def revoke_approval(self, name: str) -> None:
        """Revoke approval for a tool (after execution, to prevent reuse)."""
        self._approved_tools.discard(name)

    def register(self, spec: ToolSpec, fn: Callable[..., Awaitable]) -> None:
        """Register a tool with its spec and implementation function.

        Raises ValueError if a tool with the same name is already registered.
        """
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        logger.info(
            "TOOL_REGISTERED name=%s timeout=%ss idempotency=%s egress=%s",
            spec.name,
            spec.timeout_seconds,
            spec.idempotency_required,
            len(spec.allowed_egress),
        )
        self._tools[spec.name] = (spec, fn)

    def get_spec(self, name: str) -> ToolSpec | None:
        """Return the ToolSpec for a registered tool, or None if not found."""
        entry = self._tools.get(name)
        return entry[0] if entry else None

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

    async def call(self, name: str, args: dict) -> dict:
        """Look up tool, validate args, run with timeout, validate result, return.

        Raises ToolNotFoundError if the tool is not registered.
        Raises SchemaValidationError if input or output fails validation.
        Raises ToolTimeoutError if the tool exceeds its timeout.
        Raises ToolExecutionError if the tool raises an exception.
        Raises EgressDeniedError if the tool attempts an unauthorized outbound call.
        """
        entry = self._tools.get(name)
        if not entry:
            raise ToolNotFoundError(f"Unknown tool: {name}")

        spec, fn = entry

        # Step 0: Authorization check — tools with APPROVAL_REQUIRED must be
        # pre-approved by the workflow before they can execute.
        if spec.authorization_mode == AuthorizationMode.APPROVAL_REQUIRED:
            if spec.name not in self._approved_tools:
                logger.warning(
                    "TOOL_APPROVAL_REQUIRED name=%s risk_tier=%s — not pre-approved",
                    spec.name, spec.risk_tier.value,
                )
                raise ToolApprovalRequiredError(
                    f"Tool '{spec.name}' requires approval (risk_tier={spec.risk_tier.value}). "
                    f"Call registry.approve_tool('{spec.name}') before calling."
                )
            # Consume the approval — one tool call per approval
            self._approved_tools.discard(spec.name)

        logger.info(
            "TOOL_CALL name=%s risk_tier=%s args_validating=true",
            spec.name, spec.risk_tier.value,
        )

        # Step 1: Validate input against the registered Pydantic model
        validated_input = validate_input(spec.input_model, args)

        # Step 2: Create SandboxedHTTPClient if the tool has an egress allowlist
        egress_filter = EgressFilter(spec.allowed_egress) if spec.allowed_egress else None

        # Step 3: Execute with timeout enforcement
        # If the tool has an egress allowlist, pass the SandboxedHTTPClient
        # as a keyword argument so the tool can use it for outbound calls.
        call_kwargs = validated_input.model_dump()
        if egress_filter is not None:
            call_kwargs["http_client"] = SandboxedHTTPClient(egress_filter)

        result = await run_with_timeout(
            fn,
            call_kwargs,
            spec.timeout_seconds,
        )

        # Step 4: Enforce output size cap (before validation)
        result = enforce_output_cap(result, spec.max_output_bytes)

        # Step 5: Validate output against the registered Pydantic model
        validated_output = validate_output(spec.output_model, result)
        logger.info(
            "TOOL_CALL_COMPLETE name=%s output_validated=true",
            name,
        )

        return validated_output.model_dump()


# Singleton instance — agents register their tools into this at import time
registry = ToolRegistry()
