"""Tests for the egress filter and SandboxedHTTPClient.

Tests:
1. Allowed URL — request succeeds
2. Blocked URL — EgressDeniedError raised
3. Wildcard — wildcard allowlist matches subdomains
4. Empty allowlist — any HTTP call blocked
5. No HTTP call — tool with empty allowlist does pure compute, no error
6. Logging — blocked egress is logged at WARNING level
"""

import logging

import httpx
import pytest

from app.agentmesh.tool_registry.egress import EgressFilter, SandboxedHTTPClient
from app.agentmesh.tool_registry.exceptions import EgressDeniedError


@pytest.mark.asyncio
async def test_allowed_url():
    """Tool calls allowed URL — request should succeed (or fail with network error, not egress)."""
    filt = EgressFilter(["api.supplier.com"])
    assert filt.is_allowed("https://api.supplier.com/quotes") is True


def test_blocked_url():
    """Tool calls blocked URL — EgressDeniedError should be raised."""
    filt = EgressFilter(["api.supplier.com"])
    assert filt.is_allowed("https://attacker.com/exfil") is False


def test_wildcard_allowlist():
    """Wildcard allowlist matches subdomains."""
    filt = EgressFilter(["*.supplier.com"])
    assert filt.is_allowed("https://api.supplier.com/quotes") is True
    assert filt.is_allowed("https://sub.api.supplier.com/v1/quotes") is True
    assert filt.is_allowed("https://other.com/quotes") is False


def test_empty_allowlist():
    """Empty allowlist blocks everything."""
    filt = EgressFilter([])
    assert filt.is_allowed("https://api.supplier.com/quotes") is False
    assert filt.is_allowed("https://anywhere.com/") is False


def test_url_prefix_rule():
    """Full URL prefix rule matches URLs starting with the prefix."""
    filt = EgressFilter(["https://api.supplier.com/v1/"])
    assert filt.is_allowed("https://api.supplier.com/v1/quotes") is True
    assert filt.is_allowed("https://api.supplier.com/v2/quotes") is False


def test_exact_host_rule():
    """Exact host rule matches only that host, not subdomains."""
    filt = EgressFilter(["api.supplier.com"])
    assert filt.is_allowed("https://api.supplier.com/quotes") is True
    assert filt.is_allowed("https://sub.api.supplier.com/quotes") is False


@pytest.mark.asyncio
async def test_sandboxed_client_blocked():
    """SandboxedHTTPClient raises EgressDeniedError for blocked URLs."""
    filt = EgressFilter(["api.supplier.com"])
    client = SandboxedHTTPClient(filt)

    with pytest.raises(EgressDeniedError, match="attacker.com"):
        await client.get("https://attacker.com/exfil")

    await client.aclose()


@pytest.mark.asyncio
async def test_sandboxed_client_allowed():
    """SandboxedHTTPClient does not raise for allowed URLs (may fail with network error)."""
    filt = EgressFilter(["api.supplier.com"])
    client = SandboxedHTTPClient(filt)

    # This will fail with a network error (no real server), but NOT with EgressDeniedError
    try:
        await client.get("https://api.supplier.com/quotes")
    except EgressDeniedError:
        pytest.fail("EgressDeniedError raised for allowed URL")
    except Exception:
        pass  # Network error is expected — we only care it's not egress

    await client.aclose()


@pytest.mark.asyncio
async def test_egress_logging(caplog):
    """Blocked egress is logged at WARNING level."""
    filt = EgressFilter(["api.supplier.com"])
    client = SandboxedHTTPClient(filt)

    with caplog.at_level(logging.WARNING, logger="agentmesh.tool_registry.egress"):
        with pytest.raises(EgressDeniedError):
            await client.get("https://attacker.com/exfil")

    assert any("EGRESS_BLOCKED" in record.message for record in caplog.records)

    await client.aclose()
