"""The Tool Registry speaks real MCP (JSON-RPC tools/list + tools/call).

Uses the official MCP Python SDK client against the server in-process.
"""
import pytest
from mcp import Client

import app.agents.sourcing_agent  # noqa: F401 — registers the sourcing tools
from app.agentmesh.mcp_server import build_mcp_server


@pytest.mark.asyncio
async def test_tools_list_exposes_read_only_tools_with_json_schemas():
    async with Client(build_mcp_server()) as client:
        result = await client.list_tools()
    tools = {t.name: t for t in result.tools}
    assert {"query_suppliers", "get_price_quote", "check_seller_rating"} <= set(tools)
    # Side-effect tools need human approval → never exposed over MCP.
    assert "create_purchase_order" not in tools and "initiate_payment" not in tools
    schema = tools["query_suppliers"].input_schema
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"item", "budget", "quantity"}
    assert tools["query_suppliers"].description


@pytest.mark.asyncio
async def test_tools_call_runs_through_the_registry():
    async with Client(build_mcp_server()) as client:
        result = await client.call_tool("query_suppliers", {"item": "USB-C cable", "budget": 3.0, "quantity": 10})
    assert result.is_error is False
    names = {s["name"] for s in result.structured_content["suppliers"]}
    assert names == {"SupplierAlpha", "SupplierBeta"}


@pytest.mark.asyncio
async def test_registry_validation_errors_come_back_as_tool_errors():
    async with Client(build_mcp_server()) as client:
        result = await client.call_tool("query_suppliers", {"item": "USB-C cable", "budget": -1, "quantity": 10})
    assert result.is_error is True
    assert "SchemaValidationError" in result.content[0].text


@pytest.mark.asyncio
async def test_side_effect_tool_cannot_be_called_over_mcp():
    async with Client(build_mcp_server()) as client:
        with pytest.raises(Exception):
            await client.call_tool("create_purchase_order", {
                "supplier_name": "SupplierAlpha", "item": "USB-C cable", "quantity": 1, "unit_price": 2.5})
