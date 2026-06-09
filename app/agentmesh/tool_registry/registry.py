import logging
from collections.abc import Awaitable, Callable

from app.agentmesh.tool_registry.exceptions import ToolNotFoundError
from app.agentmesh.tool_registry.sandbox import run_with_timeout
from app.agentmesh.tool_registry.schema_validation import validate_input, validate_output
from app.agentmesh.tool_registry.spec import ToolSpec

logger = logging.getLogger("agentmesh.tool_registry")


class ToolRegistry:
    """Generic tool registry — the single harness layer with no off-the-shelf equivalent.

    Any agent's LangGraph nodes call through this registry to invoke tools.
    The registry handles:
    1. Input schema validation (before execution)
    2. Timeout enforcement (during execution)
    3. Output schema validation (after execution)

    The registry contains ZERO agent-specific code. Tools register themselves;
    the registry just enforces the contract.
    """

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, Callable[..., Awaitable]]] = {}

    def register(self, spec: ToolSpec, fn: Callable[..., Awaitable]) -> None:
        """Register a tool with its spec and implementation function.

        Raises ValueError if a tool with the same name is already registered.
        """
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        logger.info(
            "TOOL_REGISTERED name=%s timeout=%ss idempotency=%s",
            spec.name,
            spec.timeout_seconds,
            spec.idempotency_required,
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
        """
        entry = self._tools.get(name)
        if not entry:
            raise ToolNotFoundError(f"Unknown tool: {name}")

        spec, fn = entry

        # Step 1: Validate input against the registered Pydantic model
        validated_input = validate_input(spec.input_model, args)
        logger.info("TOOL_CALL name=%s args_validated=true", name)

        # Step 2: Execute with timeout enforcement
        result = await run_with_timeout(
            fn,
            validated_input.model_dump(),
            spec.timeout_seconds,
        )

        # Step 3: Validate output against the registered Pydantic model
        validated_output = validate_output(spec.output_model, result)
        logger.info(
            "TOOL_CALL_COMPLETE name=%s output_validated=true",
            name,
        )

        return validated_output.model_dump()


# Singleton instance — agents register their tools into this at import time
registry = ToolRegistry()
