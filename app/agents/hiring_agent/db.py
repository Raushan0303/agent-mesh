"""Database access layer for the hiring agent.

Same pattern as sourcing_agent/db.py: a connection pool to the shared
'agentmesh' Postgres database plus an idempotency table for the one
side-effecting tool this agent has — sending an offer.
"""

import logging

import asyncpg

from app.core.config import settings

logger = logging.getLogger("agentmesh.hiring_agent.db")

_pool: asyncpg.Pool | None = None


async def get_db_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            settings.agent_db_dsn,
            min_size=2,
            max_size=10,
        )
        logger.info("DB_POOL_CREATED dsn=%s", settings.agent_db_dsn)
    return _pool


async def close_db_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("DB_POOL_CLOSED")


async def init_tables() -> None:
    """Create the idempotency table if it doesn't exist."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS hiring_agent_offers (
                idempotency_key   TEXT PRIMARY KEY,
                offer_id          TEXT NOT NULL,
                candidate_name    TEXT NOT NULL,
                role              TEXT NOT NULL,
                offer_amount      DOUBLE PRECISION NOT NULL,
                status            TEXT NOT NULL DEFAULT 'sent',
                created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        logger.info("DB_TABLES_INITIALIZED")


async def clear_tables() -> None:
    """Clear all idempotency rows (for tests)."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM hiring_agent_offers")


async def count_offers() -> int:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT count(*) FROM hiring_agent_offers")


async def count_offers_for_workflow(workflow_id: str) -> int:
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM hiring_agent_offers WHERE idempotency_key LIKE $1",
            f"{workflow_id}:%",
        )
