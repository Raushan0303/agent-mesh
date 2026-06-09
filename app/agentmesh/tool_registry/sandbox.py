import asyncio
import logging
from collections.abc import Awaitable, Callable

from app.agentmesh.tool_registry.exceptions import ToolExecutionError, ToolTimeoutError

logger = logging.getLogger("agentmesh.tool_registry.sandbox")


async def run_with_timeout(
    fn: Callable[..., Awaitable],
    args: dict,
    timeout_seconds: float,
) -> dict:
    """Run an async tool function with timeout enforcement.

    If the function exceeds timeout_seconds, ToolTimeoutError is raised.
    If the function raises any other exception, ToolExecutionError is raised.
    """
    try:
        result = await asyncio.wait_for(
            fn(**args),
            timeout=timeout_seconds,
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(
            "TOOL_TIMEOUT fn=%s timeout=%ss — killing execution",
            fn.__name__,
            timeout_seconds,
        )
        raise ToolTimeoutError(
            f"Tool {fn.__name__} exceeded timeout of {timeout_seconds}s"
        )
    except Exception as e:
        logger.warning(
            "TOOL_EXECUTION_ERROR fn=%s error=%s",
            fn.__name__,
            str(e),
        )
        raise ToolExecutionError(
            f"Tool {fn.__name__} raised exception: {e}"
        ) from e
