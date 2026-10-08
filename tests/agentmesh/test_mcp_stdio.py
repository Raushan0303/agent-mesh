"""The MCP server also runs over stdio (how desktop clients / IDEs launch it)."""
import sys

import pytest
from mcp import Client, StdioServerParameters


@pytest.mark.asyncio
async def test_stdio_transport_lists_and_calls_tools():
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.agentmesh.mcp_server"])
    async with Client(params) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        result = await client.call_tool("check_seller_rating", {"supplier_name": "SupplierAlpha"})
    assert "check_seller_rating" in tools
    assert result.structured_content["supplier_name"] == "SupplierAlpha"
