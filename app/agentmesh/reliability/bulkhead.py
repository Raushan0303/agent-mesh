"""Bulkhead — limits concurrent in-flight calls to a downstream.

The bulkhead pattern isolates failure domains: if one downstream is slow,
it can't consume all Worker capacity and starve other Workflows. This
sits on top of (not instead of) Temporal's own retry policy.

Use case: if a downstream tool is slow, without a bulkhead it could occupy
all Worker activity slots, causing other tools' calls in other Workflows
to time out waiting for a slot. The bulkhead caps how many concurrent
calls to a given downstream can run at once.
"""

import asyncio
import logging

logger = logging.getLogger("agentmesh.reliability.bulkhead")


class BulkheadFullError(Exception):
    """Raised when the bulkhead is at capacity."""

    def __init__(self, name: str, max_concurrent: int):
        self.name = name
        self.max_concurrent = max_concurrent
        super().__init__(
            f"Bulkhead '{name}' at capacity ({max_concurrent} concurrent calls) — "
            f"rejecting call to protect other Workflows"
        )


class Bulkhead:
    """Limits concurrent in-flight calls to a downstream.

    Args:
        name: Identifier for this bulkhead (e.g., the downstream tool name).
        max_concurrent: Maximum concurrent calls allowed.
    """

    def __init__(self, name: str, max_concurrent: int = 10):
        self.name = name
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active = 0

    @property
    def active_count(self) -> int:
        return self._active

    @property
    def available(self) -> int:
        return self.max_concurrent - self._active

    async def acquire(self) -> None:
        """Acquire a slot. Raises BulkheadFullError if at capacity (non-blocking)."""
        if self._active >= self.max_concurrent:
            logger.warning(
                "BULKHEAD_FULL name=%s active=%d max=%d",
                self.name,
                self._active,
                self.max_concurrent,
            )
            raise BulkheadFullError(self.name, self.max_concurrent)
        await self._semaphore.acquire()
        self._active += 1
        logger.debug(
            "BULKHEAD_ACQUIRE name=%s active=%d/%d",
            self.name,
            self._active,
            self.max_concurrent,
        )

    def release(self) -> None:
        """Release a slot."""
        if self._active > 0:
            self._active -= 1
            self._semaphore.release()
            logger.debug(
                "BULKHEAD_RELEASE name=%s active=%d/%d",
                self.name,
                self._active,
                self.max_concurrent,
            )

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        self.release()


# ── Global registry of bulkheads, keyed by downstream name ──

_bulkheads: dict[str, Bulkhead] = {}


def get_bulkhead(name: str, max_concurrent: int = 10) -> Bulkhead:
    """Get or create a bulkhead for a downstream."""
    if name not in _bulkheads:
        _bulkheads[name] = Bulkhead(name=name, max_concurrent=max_concurrent)
    return _bulkheads[name]


def reset_all_bulkheads() -> None:
    """Reset all bulkheads (for testing)."""
    _bulkheads.clear()


def get_all_bulkhead_states() -> dict[str, dict]:
    """Get the state of all bulkheads (for monitoring)."""
    return {
        name: {
            "active": b.active_count,
            "max": b.max_concurrent,
            "available": b.available,
        }
        for name, b in _bulkheads.items()
    }
