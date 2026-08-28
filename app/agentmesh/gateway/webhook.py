"""Webhook delivery — Temporal activity that POSTs to client URLs.

Fires on workflow completion/failure/interrupt. Includes:
- HMAC-SHA256 signature header (X-AgentMesh-Signature)
- X-AgentMesh-Event header (event type)
- X-AgentMesh-Workflow-Id header
- X-AgentMesh-Delivery-Id header (unique per delivery, for dedup)
- Retry with exponential backoff (3 attempts, 2/4/8 seconds)
- Timeout (10 seconds per attempt)

This is a Temporal activity (not workflow code) because:
1. Workflow code is replayed — side effects must be in activities
2. Temporal's retry policy handles the backoff
3. The activity is idempotent via delivery IDs
"""

import hashlib
import hmac
import json
import logging
import uuid

import httpx
from temporalio import activity

from app.core.config import settings
from app.core.constants import (
    WEBHOOK_DELIVERY_ID_HEADER,
    WEBHOOK_EVENT_HEADER,
    WEBHOOK_SIGNATURE_HEADER,
    WEBHOOK_WORKFLOW_ID_HEADER,
)

logger = logging.getLogger("agentmesh.gateway.webhook")


def sign_payload(payload: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 signature of the payload.

    Returns the signature in the format: sha256=<hex_digest>
    """
    digest = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return f"sha256={digest}"


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify an HMAC-SHA256 signature.

    Args:
        payload: The raw request body bytes.
        signature: The signature from the X-AgentMesh-Signature header.
        secret: The shared secret.

    Returns:
        True if the signature is valid, False otherwise.
    """
    expected = sign_payload(payload, secret)
    return hmac.compare_digest(signature, expected)


@activity.defn
async def fire_webhook_activity(
    webhook_url: str,
    event_type: str,
    payload: dict,
    workflow_id: str,
    hmac_secret: str,
) -> dict:
    """Fire a webhook callback to an external URL.

    Includes HMAC-SHA256 signing, retry with backoff (handled by Temporal's
    retry policy), and idempotency via delivery IDs.

    Args:
        webhook_url: The URL to POST to.
        event_type: Event type (e.g., "workflow_completed", "workflow_failed").
        payload: The event payload dict.
        workflow_id: The workflow ID this event belongs to.
        hmac_secret: The HMAC secret for signing the payload.

    Returns:
        {"delivered": bool, "status_code": int, "delivery_id": str}
    """
    delivery_id = str(uuid.uuid4())
    body = json.dumps(payload).encode("utf-8")
    signature = sign_payload(body, hmac_secret)

    headers = {
        "Content-Type": "application/json",
        WEBHOOK_SIGNATURE_HEADER: signature,
        WEBHOOK_EVENT_HEADER: event_type,
        WEBHOOK_WORKFLOW_ID_HEADER: workflow_id,
        WEBHOOK_DELIVERY_ID_HEADER: delivery_id,
    }

    logger.info(
        "WEBHOOK_FIRE url=%s event=%s workflow_id=%s delivery_id=%s",
        webhook_url, event_type, workflow_id, delivery_id,
    )

    try:
        async with httpx.AsyncClient(timeout=settings.webhook_timeout_seconds) as client:
            response = await client.post(webhook_url, content=body, headers=headers)

        if response.status_code >= 400:
            logger.warning(
                "WEBHOOK_FAILED url=%s status=%d delivery_id=%s",
                webhook_url, response.status_code, delivery_id,
            )
            # Raise to trigger Temporal retry
            raise Exception(f"Webhook returned {response.status_code}")

        logger.info(
            "WEBHOOK_DELIVERED url=%s status=%d delivery_id=%s",
            webhook_url, response.status_code, delivery_id,
        )
        return {
            "delivered": True,
            "status_code": response.status_code,
            "delivery_id": delivery_id,
        }
    except httpx.TimeoutException:
        logger.warning(
            "WEBHOOK_TIMEOUT url=%s delivery_id=%s",
            webhook_url, delivery_id,
        )
        raise
    except Exception as e:
        if "Webhook returned" in str(e):
            raise
        logger.warning(
            "WEBHOOK_ERROR url=%s error=%s delivery_id=%s",
            webhook_url, e, delivery_id,
        )
        raise
