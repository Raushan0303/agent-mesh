"""LangGraph Postgres checkpointer for the hiring agent.

Same rationale as sourcing_agent/checkpointer.py: the graph's state
(screening notes, interview transcript, offer decision) is persisted
independently of Temporal's Event History, via its own AsyncPostgresSaver
pool. This is what makes interrupt() at the Human Review node durable.
"""

import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings

logger = logging.getLogger("agentmesh.hiring_agent.checkpointer")

_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None


def _get_pool() -> AsyncConnectionPool:
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
    global _checkpointer
    if _checkpointer is None:
        pool = _get_pool()
        await pool.open()
        _checkpointer = AsyncPostgresSaver(pool)
        await _checkpointer.setup()
        logger.info("CHECKPOINT_INITIALIZED")
    return _checkpointer


async def close_checkpointer() -> None:
    global _pool, _checkpointer
    if _checkpointer is not None:
        _checkpointer = None
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("CHECKPOINT_POOL_CLOSED")
