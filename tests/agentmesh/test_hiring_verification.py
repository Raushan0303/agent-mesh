"""Tests for hiring agent external outcome verification.

Tests:
1. Offer verified — offer exists in database, all fields match
2. Offer not verified — offer not in database
3. Offer field mismatch — wrong candidate name
4. Offer field mismatch — wrong amount
5. Offer field mismatch — wrong status
6. Verification queries the correct table
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.hiring_agent.verification import verify_offer_sent


def _make_mock_pool(fetchrow_return):
    """Create a mock pool that returns the given value from fetchrow."""
    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock_pool, mock_conn


@pytest.mark.asyncio
async def test_verify_offer_sent_pass():
    """Offer verified — exists in database, all fields match."""
    mock_row = {
        "offer_id": "OFFER-test123",
        "candidate_name": "Alice Chen",
        "role": "Backend Engineer",
        "offer_amount": 120000.0,
        "status": "sent",
    }
    mock_pool, mock_conn = _make_mock_pool(mock_row)

    with patch("app.agents.hiring_agent.verification.get_db_pool", AsyncMock(return_value=mock_pool)):
        result = await verify_offer_sent("OFFER-test123", "Alice Chen", "Backend Engineer", 120000.0)

    assert result["verified"] is True
    assert result["side_effect_id"] == "OFFER-test123"
    assert result["mismatches"] == []


@pytest.mark.asyncio
async def test_verify_offer_sent_not_found():
    """Offer not verified — offer not in database."""
    mock_pool, mock_conn = _make_mock_pool(None)

    with patch("app.agents.hiring_agent.verification.get_db_pool", AsyncMock(return_value=mock_pool)):
        result = await verify_offer_sent("OFFER-nonexistent", "Alice", "Backend", 120000.0)

    assert result["verified"] is False
    assert result["side_effect_id"] == "OFFER-nonexistent"
    assert "offer_not_found" in result["mismatches"]


@pytest.mark.asyncio
async def test_verify_offer_sent_wrong_candidate():
    """Offer field mismatch — candidate_name doesn't match."""
    mock_row = {
        "offer_id": "OFFER-test123",
        "candidate_name": "Bob Smith",  # wrong
        "role": "Backend Engineer",
        "offer_amount": 120000.0,
        "status": "sent",
    }
    mock_pool, mock_conn = _make_mock_pool(mock_row)

    with patch("app.agents.hiring_agent.verification.get_db_pool", AsyncMock(return_value=mock_pool)):
        result = await verify_offer_sent("OFFER-test123", "Alice Chen", "Backend Engineer", 120000.0)

    assert result["verified"] is False
    assert any("candidate_name" in m for m in result["mismatches"])


@pytest.mark.asyncio
async def test_verify_offer_sent_wrong_amount():
    """Offer field mismatch — offer_amount doesn't match (beyond tolerance)."""
    mock_row = {
        "offer_id": "OFFER-test123",
        "candidate_name": "Alice Chen",
        "role": "Backend Engineer",
        "offer_amount": 100000.0,  # wrong by $20k
        "status": "sent",
    }
    mock_pool, mock_conn = _make_mock_pool(mock_row)

    with patch("app.agents.hiring_agent.verification.get_db_pool", AsyncMock(return_value=mock_pool)):
        result = await verify_offer_sent("OFFER-test123", "Alice Chen", "Backend Engineer", 120000.0)

    assert result["verified"] is False
    assert any("offer_amount" in m for m in result["mismatches"])


@pytest.mark.asyncio
async def test_verify_offer_sent_wrong_status():
    """Offer field mismatch — status is not 'sent'."""
    mock_row = {
        "offer_id": "OFFER-test123",
        "candidate_name": "Alice Chen",
        "role": "Backend Engineer",
        "offer_amount": 120000.0,
        "status": "pending",  # wrong
    }
    mock_pool, mock_conn = _make_mock_pool(mock_row)

    with patch("app.agents.hiring_agent.verification.get_db_pool", AsyncMock(return_value=mock_pool)):
        result = await verify_offer_sent("OFFER-test123", "Alice Chen", "Backend Engineer", 120000.0)

    assert result["verified"] is False
    assert any("status" in m for m in result["mismatches"])


@pytest.mark.asyncio
async def test_verify_offer_sent_checks_correct_table():
    """Verification queries the correct table (hiring_agent_offers)."""
    mock_pool, mock_conn = _make_mock_pool(None)

    with patch("app.agents.hiring_agent.verification.get_db_pool", AsyncMock(return_value=mock_pool)):
        await verify_offer_sent("OFFER-test", "Alice", "Backend", 120000.0)

    sql_arg = mock_conn.fetchrow.call_args[0][0]
    assert "hiring_agent_offers" in sql_arg
    assert "offer_id" in sql_arg


@pytest.mark.asyncio
async def test_verify_offer_sent_unknown_on_db_error():
    """Offer verification returns UNKNOWN when the DB query fails — NOT safe to retry."""
    mock_pool = MagicMock()
    mock_pool.acquire.side_effect = Exception("connection refused")

    with patch("app.agents.hiring_agent.verification.get_db_pool", AsyncMock(return_value=mock_pool)):
        result = await verify_offer_sent("OFFER-test", "Alice", "Backend", 120000.0)

    assert result["verified"] is False
    assert result["status"] == "unknown"
    assert "db_query_failed" in result.get("error", "")


@pytest.mark.asyncio
async def test_verify_offer_sent_failed_not_unknown_when_row_missing():
    """Offer not found = FAILED (safe to retry), not UNKNOWN."""
    mock_pool, mock_conn = _make_mock_pool(None)

    with patch("app.agents.hiring_agent.verification.get_db_pool", AsyncMock(return_value=mock_pool)):
        result = await verify_offer_sent("OFFER-nonexistent", "Alice", "Backend", 120000.0)

    assert result["verified"] is False
    assert result["status"] == "failed"
    assert "offer_not_found" in result["mismatches"]
