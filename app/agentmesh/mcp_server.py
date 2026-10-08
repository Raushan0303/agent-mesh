"""MCP server for the Tool Registry — real Model Context Protocol.

Exposes registered tools to any MCP client (Claude Desktop, IDEs, other
agents, InferRoute's MCP gateway) over the protocol's JSON-RPC methods:

  tools/list → every exposed ToolSpec, with JSON Schemas generated from the
               tool's Pydantic input/output models
  tools/call → ToolRegistry.call(), so the registry's 5 checks (input
               validation, egress allowlist, timeout, output cap, output
               validation) apply to MCP callers exactly as to agents

Transports:
  Streamable HTTP  mounted on the gateway at /mcp (see app/main.py)
  stdio            python -m app.agentmesh.mcp_server

Only READ_ONLY tools are exposed by default: side-effect tools
(create_purchase_order, initiate_payment) need a human approval that an
MCP caller cannot give.
"""

import json
import logging
from collections.abc import Callable

import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import MCPError
from mcp_types.jsonrpc import INVALID_PARAMS

from app.agentmesh.tool_registry import registry as default_registry
from app.agentmesh.tool_registry.registry import ToolRegistry
from app.agentmesh.tool_registry.spec import RiskTier, ToolSpec

logger = logging.getLogger("agentmesh.mcp")


def read_only(spec: ToolSpec) -> bool:
    return spec.risk_tier == RiskTier.READ_ONLY


def build_mcp_server(
    tool_registry: ToolRegistry = default_registry,
    expose: Callable[[ToolSpec], bool] = read_only,
) -> Server:
    def exposed() -> list[ToolSpec]:
        specs = (tool_registry.get_spec(n) for n in tool_registry.list_tools())
        return [s for s in specs if s is not None and expose(s)]

    async def on_list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[
            types.Tool(
                name=spec.name,
                description=spec.description or None,
                input_schema=spec.input_model.model_json_schema(),
                output_schema=spec.output_model.model_json_schema(),
            )
            for spec in exposed()
        ])

    async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        spec = tool_registry.get_spec(params.name)
        if spec is None or not expose(spec):
            # Unknown / hidden tool is a protocol-level error (JSON-RPC
            # -32602), not a tool error.
            raise MCPError(INVALID_PARAMS, f"Unknown tool: {params.name}")
        logger.info("MCP_TOOLS_CALL name=%s", params.name)
        try:
            result = await tool_registry.call(params.name, params.arguments or {})
        except Exception as e:
            # Tool failures go back inside the result (is_error) so the
            # calling model can see them and self-correct.
            return types.CallToolResult(
                is_error=True,
                content=[types.TextContent(text=f"{type(e).__name__}: {e}")],
            )
        return types.CallToolResult(
            content=[types.TextContent(text=json.dumps(result))],
            structured_content=result,
        )

    return Server(
        "agentmesh-tool-registry",
        version="0.1.0",
        instructions="AgentMesh Tool Registry. Read-only tools; every call is schema-validated, "
                     "time-limited and egress-filtered.",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def _run_stdio() -> None:
    from mcp.server.stdio import stdio_server

    import app.agents.sourcing_agent  # noqa: F401 — registers tools
    import app.agents.hiring_agent  # noqa: F401

    server = build_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import anyio

    anyio.run(_run_stdio)
