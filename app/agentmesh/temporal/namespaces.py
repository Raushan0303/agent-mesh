"""Temporal Namespace helper — tenant isolation primitive.

A Namespace is Temporal's isolation boundary above Task Queues. Each tenant
gets their own Namespace, and Workflows in one Namespace are invisible to
queries scoped to another Namespace. This is a platform capability — the
agent doesn't implement isolation, it just runs inside whichever Namespace
it's told to.

Convention:
- The gateway assigns a Namespace per tenant (default: "default" for single-tenant).
- The worker connects to the same Namespace as the gateway.
- Namespaces are registered via `temporal cli namespace register` or the
  auto-setup image (which creates the "default" Namespace automatically).
"""

import logging
from typing import Dict

from temporalio.client import Client

from app.core.config import settings

logger = logging.getLogger("agentmesh.temporal.namespaces")

# Cache of Namespace-scoped clients
_namespace_clients: Dict[str, Client] = {}


async def get_namespace_client(namespace: str | None = None) -> Client:
    """Get a Temporal client scoped to a specific Namespace.

    If namespace is None, uses the default namespace from settings.
    Caches clients per namespace for reuse.
    """
    ns = namespace or settings.temporal_namespace
    if ns not in _namespace_clients:
        client = await Client.connect(
            settings.temporal_host,
            namespace=ns,
        )
        _namespace_clients[ns] = client
        logger.info("NAMESPACE_CLIENT_CREATED namespace=%s", ns)
    return _namespace_clients[ns]


async def ensure_namespace_exists(namespace: str) -> None:
    """Ensure a Namespace exists in the Temporal cluster.

    For local development with auto-setup, the "default" Namespace is
    created automatically. For custom Namespaces, you need to register
    them via the Temporal CLI:
        temporal operator namespace create <namespace>
    """
    # This is a no-op for the default namespace — auto-setup handles it.
    # For custom namespaces, the operator must register them before use.
    logger.info("NAMESPACE_ENSURE namespace=%s (assumed pre-registered)", namespace)


def clear_namespace_clients() -> None:
    """Clear the client cache (for tests)."""
    _namespace_clients.clear()
