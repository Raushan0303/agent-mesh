"""Tests: Tool Registry fault injection, schema validation, and sandbox behavior.

These tests prove the registry's timeout, validation, and error handling
actually work under adversarial conditions.
"""

import asyncio

import pytest
from pydantic import BaseModel

from app.agentmesh.tool_registry import (
    SchemaValidationError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistry,
    ToolSpec,
    ToolTimeoutError,
)


# ── Test fixtures ──


class SimpleInput(BaseModel):
    value: int


class SimpleOutput(BaseModel):
    result: int


def _make_registry_with_tool(
    name: str = "test_tool",
    timeout: float = 5.0,
    impl=None,
) -> ToolRegistry:
    """Build a fresh registry with one test tool registered."""
    reg = ToolRegistry()

    async def default_impl(value: int) -> dict:
        return {"result": value * 2}

    reg.register(
        ToolSpec(
            name=name,
            input_model=SimpleInput,
            output_model=SimpleOutput,
            timeout_seconds=timeout,
            idempotency_required=False,
        ),
        impl or default_impl,
    )
    return reg


# ── 1. Fault-injection: tool hangs, timeout fires ──


@pytest.mark.asyncio
async def test_tool_timeout_fires():
    """A tool that hangs for 30s must be killed by the 2s timeout."""
    reg = _make_registry_with_tool(timeout=2.0)

    async def hanging_impl(value: int) -> dict:
        await asyncio.sleep(30)
        return {"result": value}

    reg.register(
        ToolSpec(
            name="hanging_tool",
            input_model=SimpleInput,
            output_model=SimpleOutput,
            timeout_seconds=2.0,
            idempotency_required=False,
        ),
        hanging_impl,
    )

    with pytest.raises(ToolTimeoutError) as exc_info:
        await reg.call("hanging_tool", {"value": 42})

    assert "exceeded timeout" in str(exc_info.value)
    assert "2.0s" in str(exc_info.value)


# ── 2. Schema validation: malformed input ──


@pytest.mark.asyncio
async def test_schema_validation_rejects_bad_input():
    """Calling a tool with missing required fields must raise SchemaValidationError."""
    reg = _make_registry_with_tool()

    with pytest.raises(SchemaValidationError) as exc_info:
        await reg.call("test_tool", {})

    assert "Input validation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_schema_validation_rejects_wrong_type():
    """Calling a tool with wrong types must raise SchemaValidationError."""
    reg = _make_registry_with_tool()

    with pytest.raises(SchemaValidationError):
        await reg.call("test_tool", {"value": "not_an_int"})


# ── 3. Schema validation: malformed output ──


@pytest.mark.asyncio
async def test_schema_validation_rejects_bad_output():
    """A tool that returns wrong fields must raise SchemaValidationError."""
    reg = ToolRegistry()

    async def bad_output_impl(value: int) -> dict:
        return {"wrong_field": 123}

    reg.register(
        ToolSpec(
            name="bad_output_tool",
            input_model=SimpleInput,
            output_model=SimpleOutput,
            timeout_seconds=5.0,
            idempotency_required=False,
        ),
        bad_output_impl,
    )

    with pytest.raises(SchemaValidationError) as exc_info:
        await reg.call("bad_output_tool", {"value": 42})

    assert "Output validation failed" in str(exc_info.value)


# ── 4. Tool not found ──


@pytest.mark.asyncio
async def test_tool_not_found():
    """Calling an unregistered tool must raise ToolNotFoundError."""
    reg = _make_registry_with_tool()

    with pytest.raises(ToolNotFoundError) as exc_info:
        await reg.call("nonexistent_tool", {"value": 1})

    assert "Unknown tool" in str(exc_info.value)


# ── 5. Tool execution error ──


@pytest.mark.asyncio
async def test_tool_execution_error():
    """A tool that raises an exception must be wrapped in ToolExecutionError."""
    reg = ToolRegistry()

    async def crashing_impl(value: int) -> dict:
        raise RuntimeError("Intentional crash")

    reg.register(
        ToolSpec(
            name="crashing_tool",
            input_model=SimpleInput,
            output_model=SimpleOutput,
            timeout_seconds=5.0,
            idempotency_required=False,
        ),
        crashing_impl,
    )

    with pytest.raises(ToolExecutionError) as exc_info:
        await reg.call("crashing_tool", {"value": 1})

    assert "Intentional crash" in str(exc_info.value)


# ── 6. Duplicate registration ──


def test_duplicate_registration_raises():
    """Registering a tool with a name that already exists must raise ValueError."""
    reg = _make_registry_with_tool(name="my_tool")

    async def another_impl(value: int) -> dict:
        return {"result": value}

    with pytest.raises(ValueError, match="Tool already registered: my_tool"):
        reg.register(
            ToolSpec(
                name="my_tool",
                input_model=SimpleInput,
                output_model=SimpleOutput,
                timeout_seconds=5.0,
                idempotency_required=False,
            ),
            another_impl,
        )


# ── 7. Successful call returns validated output ──


@pytest.mark.asyncio
async def test_successful_call_returns_validated_output():
    """A normal tool call must return the validated output as a dict."""
    reg = _make_registry_with_tool()

    result = await reg.call("test_tool", {"value": 21})

    assert result == {"result": 42}


# ── 8. Sandbox: tool that writes to filesystem (logged, not blocked in Week 2) ──


@pytest.mark.asyncio
async def test_sandbox_filesystem_write_is_executed_but_caught():
    """Week 2: filesystem writes are not blocked (full sandbox is Week 3).

    The tool tries to write to /tmp — it succeeds (Week 2 doesn't block
    filesystem access), but the registry catches any exception and wraps
    it in ToolExecutionError. If the write succeeds, the tool returns
    normally with the validated output.
    """
    import os

    reg = ToolRegistry()

    class WriteInput(BaseModel):
        path: str
        content: str

    class WriteOutput(BaseModel):
        written: bool
        path: str

    async def write_impl(path: str, content: str) -> dict:
        with open(path, "w") as f:
            f.write(content)
        return {"written": True, "path": path}

    reg.register(
        ToolSpec(
            name="write_tool",
            input_model=WriteInput,
            output_model=WriteOutput,
            timeout_seconds=5.0,
            idempotency_required=False,
        ),
        write_impl,
    )

    test_path = "/tmp/sandbox_escape_test_w2"
    result = await reg.call(
        "write_tool", {"path": test_path, "content": "test"}
    )

    assert result["written"] is True
    assert os.path.exists(test_path)

    # Cleanup
    os.remove(test_path)


@pytest.mark.asyncio
async def test_approval_required_blocks_unapproved_tool():
    """A tool with authorization_mode=APPROVAL_REQUIRED raises if not pre-approved."""
    from app.agentmesh.tool_registry.exceptions import ToolApprovalRequiredError
    from app.agentmesh.tool_registry.spec import AuthorizationMode, RiskTier

    reg = ToolRegistry()

    class In(BaseModel):
        x: int

    class Out(BaseModel):
        result: int

    async def impl(x: int) -> dict:
        return {"result": x * 2}

    reg.register(
        ToolSpec(
            name="dangerous_tool",
            input_model=In,
            output_model=Out,
            timeout_seconds=5.0,
            idempotency_required=True,
            risk_tier=RiskTier.IRREVERSIBLE,
            authorization_mode=AuthorizationMode.APPROVAL_REQUIRED,
        ),
        impl,
    )

    # Without approval — should raise
    with pytest.raises(ToolApprovalRequiredError):
        await reg.call("dangerous_tool", {"x": 5})

    # With approval — should succeed
    reg.approve_tool("dangerous_tool")
    result = await reg.call("dangerous_tool", {"x": 5})
    assert result["result"] == 10

    # Approval is consumed — second call without re-approval should raise
    with pytest.raises(ToolApprovalRequiredError):
        await reg.call("dangerous_tool", {"x": 5})


@pytest.mark.asyncio
async def test_automatic_tool_runs_without_approval():
    """A tool with authorization_mode=AUTOMATIC runs without pre-approval."""
    from app.agentmesh.tool_registry.spec import RiskTier

    reg = ToolRegistry()

    class In(BaseModel):
        x: int

    class Out(BaseModel):
        result: int

    async def impl(x: int) -> dict:
        return {"result": x}

    reg.register(
        ToolSpec(
            name="safe_tool",
            input_model=In,
            output_model=Out,
            timeout_seconds=5.0,
            idempotency_required=False,
            risk_tier=RiskTier.READ_ONLY,
        ),
        impl,
    )

    result = await reg.call("safe_tool", {"x": 42})
    assert result["result"] == 42
