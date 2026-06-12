"""LangGraph Postgres checkpointer for the sourcing agent.

This gives the LangGraph graph its own persistence layer, separate from
Temporal's Event History. The graph's state (suppliers, scores, decisions)
survives a Worker kill independently — even if Temporal's Activity hasn't
completed, the graph's checkpoint is in Postgres.

Key difference from Temporal's Event History:
- Temporal records every Workflow/Activity event (durable execution log)
- LangGraph's checkpointer records the graph's state at each superstep
  (snapshot of the TypedDict state between nodes)
- Both are needed: Temporal owns durability, LangGraph owns graph state

Uses AsyncPostgresSaver (not the sync PostgresSaver) because the graph
is invoked with ainvoke() inside an async Temporal Activity.
"""

import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings

logger = logging.getLogger("agentmesh.sourcing_agent.checkpointer")

_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None


def _get_pool() -> AsyncConnectionPool:
    """Get or create the async psycopg connection pool."""
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            conninfo=settings.agent_db_dsn,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        logger.info("CHECKPOINT_POOL_CREATED dsn=%s", settings.agent_db_dsn)
    return _pool


async def get_checkpointer() -> AsyncPostgresSaver:
    """Get or create the AsyncPostgresSaver checkpointer.

    The checkpointer is used by LangGraph to persist graph state between
    supersteps. This is what makes interrupt() durable — the graph's state
    at the Approve node is saved to Postgres, so even if the Worker dies
    while paused, the graph resumes at Approve (not Research) after restart.
    """
    global _checkpointer
    if _checkpointer is None:
        pool = _get_pool()
        await pool.open()
        _checkpointer = AsyncPostgresSaver(pool)
        await _checkpointer.setup()
        logger.info("CHECKPOINT_INITIALIZED")
    return _checkpointer


async def close_checkpointer() -> None:
    """Close the checkpointer and pool (for tests and clean shutdown)."""
    global _pool, _checkpointer
    if _checkpointer is not None:
        _checkpointer = None
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("CHECKPOINT_POOL_CLOSED")
