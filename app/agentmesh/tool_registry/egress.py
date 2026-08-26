"""Egress filter — URL allowlist for outbound tool calls.

Tools must use the SandboxedHTTPClient (which wraps httpx with this
filter). The filter checks every request URL against the allowlist.
URLs not on the allowlist are blocked and logged.

Allowlist entries can be:
- Exact host: "api.supplier.com"
- Wildcard host: "*.supplier.com"
- Full URL prefix: "https://api.supplier.com/v1/"
"""

import logging
from urllib.parse import urlparse

import httpx

from app.agentmesh.tool_registry.exceptions import EgressDeniedError

logger = logging.getLogger("agentmesh.tool_registry.egress")


class EgressFilter:
    """URL allowlist for outbound tool calls.

    Tools should use the SandboxedHTTPClient (which wraps httpx with
    this filter). The filter checks every request URL against the
    allowlist. URLs not on the allowlist are blocked.

    Allowlist entries can be:
    - Exact host: "api.supplier.com"
    - Wildcard host: "*.supplier.com"
    - Full URL prefix: "https://api.supplier.com/v1/"
    """

    def __init__(self, allowlist: list[str]):
        self._allowlist = allowlist
        self._compiled = [self._compile_rule(r) for r in allowlist]

    def is_allowed(self, url: str) -> bool:
        """Check if a URL is allowed under the allowlist."""
        parsed = urlparse(url)
        host = parsed.hostname or ""
        for rule in self._compiled:
            if rule(host, parsed.path, url):
                return True
        return False

    def _compile_rule(self, rule: str):
        """Compile a rule string into a predicate function."""
        if rule.startswith("*."):
            suffix = rule[1:]  # ".supplier.com"
            return lambda host, path, url: host.endswith(suffix)
        elif rule.startswith("http"):
            return lambda host, path, url: url.startswith(rule)
        else:
            return lambda host, path, url: host == rule


class SandboxedHTTPClient(httpx.AsyncClient):
    """httpx client with egress filtering.

    Tools should use this instead of a plain httpx.AsyncClient.
    The egress filter checks every request URL against the allowlist.

    Usage:
        client = SandboxedHTTPClient(egress_filter)
        response = await client.get("https://api.supplier.com/quotes")
        # If api.supplier.com is not on the allowlist, raises EgressDeniedError
    """

    def __init__(self, egress_filter: EgressFilter, **kwargs):
        super().__init__(**kwargs)
        self._egress = egress_filter

    async def request(self, method: str, url: str, **kwargs):
        if not self._egress.is_allowed(str(url)):
            logger.warning(
                "EGRESS_BLOCKED url=%s — not in allowlist",
                url,
            )
            raise EgressDeniedError(
                f"Outbound request to {url} blocked by egress filter"
            )
        return await super().request(method, url, **kwargs)
