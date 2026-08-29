"""Tests for external outcome verification.

Tests:
1. PO verified — PO exists in database, verification passes
2. PO not verified — PO not in database, verification fails
3. Payment verified — payment exists in database, verification passes
4. Payment not verified — payment not in database, verification fails
5. Verification queries correct table (PO)
6. Verification queries correct table (payment)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.sourcing_agent.verification import (
    verify_payment_initiated,
    verify_po_exists,
)


def _make_mock_pool(fetchrow_return):
    """Create a mock pool that returns the given value from fetchrow."""
    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock_pool, mock_conn


@pytest.mark.asyncio
async def test_verify_po_exists_pass():
    """PO verified — PO exists in database, verification passes."""
    mock_row = {
        "po_id": "PO-test123",
        "status": "created",
        "supplier_name": "SupplierAlpha",
        "item": "USB-C cable",
        "quantity": 10,
        "unit_price": 2.50,
    }
    mock_pool, mock_conn = _make_mock_pool(mock_row)

    with patch("app.agents.sourcing_agent.verification.get_db_pool", AsyncMock(return_value=mock_pool)):
        result = await verify_po_exists("PO-test123")

    assert result["verified"] is True
    assert result["po_id"] == "PO-test123"
    assert result["po_status"] == "created"
    assert result["supplier_name"] == "SupplierAlpha"


@pytest.mark.asyncio
async def test_verify_po_exists_fail():
    """PO not verified — PO not in database, verification fails."""
    mock_pool, mock_conn = _make_mock_pool(None)

    with patch("app.agents.sourcing_agent.verification.get_db_pool", AsyncMock(return_value=mock_pool)):
        result = await verify_po_exists("PO-nonexistent")

    assert result["verified"] is False
    assert result["po_id"] == "PO-nonexistent"
    assert result["po_status"] is None


@pytest.mark.asyncio
async def test_verify_payment_initiated_pass():
    """Payment verified — payment exists in database, verification passes."""
    mock_row = {
        "payment_id": "PAY-test123",
        "po_id": "PO-test123",
        "status": "initiated",
        "amount": 25.00,
    }
    mock_pool, mock_conn = _make_mock_pool(mock_row)

    with patch("app.agents.sourcing_agent.verification.get_db_pool", AsyncMock(return_value=mock_pool)):
        result = await verify_payment_initiated("PAY-test123")

    assert result["verified"] is True
    assert result["payment_id"] == "PAY-test123"
    assert result["payment_status"] == "initiated"
    assert result["po_id"] == "PO-test123"


@pytest.mark.asyncio
async def test_verify_payment_initiated_fail():
    """Payment not verified — payment not in database, verification fails."""
    mock_pool, mock_conn = _make_mock_pool(None)

    with patch("app.agents.sourcing_agent.verification.get_db_pool", AsyncMock(return_value=mock_pool)):
        result = await verify_payment_initiated("PAY-nonexistent")

    assert result["verified"] is False
    assert result["payment_id"] == "PAY-nonexistent"
    assert result["payment_status"] is None


@pytest.mark.asyncio
async def test_verify_po_exists_checks_correct_table():
    """Verification queries the correct table (sourcing_agent_purchase_orders)."""
    mock_pool, mock_conn = _make_mock_pool(None)

    with patch("app.agents.sourcing_agent.verification.get_db_pool", AsyncMock(return_value=mock_pool)):
        await verify_po_exists("PO-test")

    sql_arg = mock_conn.fetchrow.call_args[0][0]
    assert "sourcing_agent_purchase_orders" in sql_arg
    assert "po_id" in sql_arg


@pytest.mark.asyncio
async def test_verify_payment_initiated_checks_correct_table():
    """Verification queries the correct table (sourcing_agent_payment_intents)."""
    mock_pool, mock_conn = _make_mock_pool(None)

    with patch("app.agents.sourcing_agent.verification.get_db_pool", AsyncMock(return_value=mock_pool)):
        await verify_payment_initiated("PAY-test")

    sql_arg = mock_conn.fetchrow.call_args[0][0]
    assert "sourcing_agent_payment_intents" in sql_arg
    assert "payment_id" in sql_arg
