"""Tests for webhook delivery and HMAC signing.

Tests:
1. Basic webhook — fires and delivers successfully
2. HMAC verification — signature is valid
3. Event filtering — only configured events fire
4. Retry — failed webhook raises (Temporal retries)
5. Dedup — delivery IDs are unique
6. Interrupt webhook — fires on interrupt events
"""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.agentmesh.gateway.webhook import (
    fire_webhook_activity,
    sign_payload,
    verify_signature,
)
from app.core.constants import (
    WEBHOOK_DELIVERY_ID_HEADER,
    WEBHOOK_EVENT_HEADER,
    WEBHOOK_SIGNATURE_HEADER,
    WEBHOOK_WORKFLOW_ID_HEADER,
)


def test_sign_payload():
    """HMAC signing — produces correct signature format."""
    payload = b'{"status": "completed"}'
    secret = "test-secret"
    sig = sign_payload(payload, secret)

    assert sig.startswith("sha256=")
    # Verify the signature is correct
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    assert sig == f"sha256={expected}"


def test_verify_signature_valid():
    """HMAC verification — valid signature returns True."""
    payload = b'{"status": "completed"}'
    secret = "test-secret"
    sig = sign_payload(payload, secret)
    assert verify_signature(payload, sig, secret) is True


def test_verify_signature_invalid():
    """HMAC verification — invalid signature returns False."""
    payload = b'{"status": "completed"}'
    secret = "test-secret"
    wrong_sig = "sha256=wrong"
    assert verify_signature(payload, wrong_sig, secret) is False


def test_verify_signature_wrong_secret():
    """HMAC verification — wrong secret returns False."""
    payload = b'{"status": "completed"}'
    sig = sign_payload(payload, "correct-secret")
    assert verify_signature(payload, sig, "wrong-secret") is False


@pytest.mark.asyncio
async def test_basic_webhook():
    """Basic webhook — fires and delivers successfully."""
    # Mock the activity context
    import app.agentmesh.gateway.webhook as webhook_mod
    original_info = webhook_mod.activity.info
    webhook_mod.activity.info = MagicMock(return_value=MagicMock(workflow_id="test-wf-123"))

    try:
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = httpx.Response(200, text="OK")

            result = await fire_webhook_activity(
                webhook_url="https://example.com/webhook",
                event_type="workflow_completed",
                payload={"status": "completed", "workflow_id": "test-wf-123"},
                workflow_id="test-wf-123",
                hmac_secret="test-secret",
            )

        assert result["delivered"] is True
        assert result["status_code"] == 200
        assert result["delivery_id"] is not None

        # Verify headers were set
        call_args = mock_post.call_args
        headers = call_args.kwargs["headers"]
        assert headers[WEBHOOK_EVENT_HEADER] == "workflow_completed"
        assert headers[WEBHOOK_WORKFLOW_ID_HEADER] == "test-wf-123"
        assert headers[WEBHOOK_DELIVERY_ID_HEADER] is not None
        assert headers[WEBHOOK_SIGNATURE_HEADER].startswith("sha256=")
    finally:
        webhook_mod.activity.info = original_info


@pytest.mark.asyncio
async def test_webhook_retry_on_failure():
    """Retry — failed webhook (4xx/5xx) raises to trigger Temporal retry."""
    import app.agentmesh.gateway.webhook as webhook_mod
    original_info = webhook_mod.activity.info
    webhook_mod.activity.info = MagicMock(return_value=MagicMock(workflow_id="test-wf-123"))

    try:
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = httpx.Response(500, text="Internal Server Error")

            with pytest.raises(Exception, match="Webhook returned 500"):
                await fire_webhook_activity(
                    webhook_url="https://example.com/webhook",
                    event_type="workflow_failed",
                    payload={"status": "failed"},
                    workflow_id="test-wf-123",
                    hmac_secret="test-secret",
                )
    finally:
        webhook_mod.activity.info = original_info


@pytest.mark.asyncio
async def test_webhook_delivery_id_unique():
    """Dedup — delivery IDs are unique across calls."""
    import app.agentmesh.gateway.webhook as webhook_mod
    original_info = webhook_mod.activity.info
    webhook_mod.activity.info = MagicMock(return_value=MagicMock(workflow_id="test-wf-123"))

    delivery_ids = []
    try:
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = httpx.Response(200, text="OK")

            for _ in range(5):
                result = await fire_webhook_activity(
                    webhook_url="https://example.com/webhook",
                    event_type="workflow_completed",
                    payload={"status": "completed"},
                    workflow_id="test-wf-123",
                    hmac_secret="test-secret",
                )
                delivery_ids.append(result["delivery_id"])

        assert len(delivery_ids) == 5
        assert len(set(delivery_ids)) == 5  # all unique
    finally:
        webhook_mod.activity.info = original_info


@pytest.mark.asyncio
async def test_webhook_hmac_correct():
    """HMAC — the signature in the header matches the payload."""
    import app.agentmesh.gateway.webhook as webhook_mod
    original_info = webhook_mod.activity.info
    webhook_mod.activity.info = MagicMock(return_value=MagicMock(workflow_id="test-wf-123"))

    try:
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = httpx.Response(200, text="OK")

            payload = {"status": "completed", "workflow_id": "test-wf-123"}
            await fire_webhook_activity(
                webhook_url="https://example.com/webhook",
                event_type="workflow_completed",
                payload=payload,
                workflow_id="test-wf-123",
                hmac_secret="my-secret",
            )

        call_args = mock_post.call_args
        headers = call_args.kwargs["headers"]
        body = call_args.kwargs["content"]

        # Verify the signature matches
        assert verify_signature(body, headers[WEBHOOK_SIGNATURE_HEADER], "my-secret") is True
    finally:
        webhook_mod.activity.info = original_info
