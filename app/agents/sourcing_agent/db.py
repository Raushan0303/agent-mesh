"""Database access layer for the sourcing agent.

Provides a connection pool to the 'agentmesh' Postgres database and
the idempotency tables used by create_purchase_order and initiate_payment.
"""

import logging

import asyncpg

from app.core.config import settings

logger = logging.getLogger("agentmesh.sourcing_agent.db")

_pool: asyncpg.Pool | None = None


async def get_db_pool() -> asyncpg.Pool:
    """Get or create the connection pool for the agentmesh database."""
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
    """Close the connection pool (for tests and clean shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("DB_POOL_CLOSED")


async def init_tables() -> None:
    """Create idempotency tables if they don't exist."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sourcing_agent_purchase_orders (
                idempotency_key   TEXT PRIMARY KEY,
                po_id             TEXT NOT NULL,
                supplier_name     TEXT NOT NULL,
                item              TEXT NOT NULL,
                quantity          INTEGER NOT NULL,
                unit_price        DOUBLE PRECISION NOT NULL,
                status            TEXT NOT NULL DEFAULT 'created',
                created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sourcing_agent_payment_intents (
                idempotency_key   TEXT PRIMARY KEY,
                payment_id        TEXT NOT NULL,
                po_id             TEXT NOT NULL,
                amount            DOUBLE PRECISION NOT NULL,
                status            TEXT NOT NULL DEFAULT 'initiated',
                created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        logger.info("DB_TABLES_INITIALIZED")


async def clear_tables() -> None:
    """Clear all idempotency rows (for tests)."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM sourcing_agent_purchase_orders")
        await conn.execute("DELETE FROM sourcing_agent_payment_intents")


async def count_purchase_orders() -> int:
    """Count total rows in the PO table."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT count(*) FROM sourcing_agent_purchase_orders")


async def count_payment_intents() -> int:
    """Count total rows in the payment table."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT count(*) FROM sourcing_agent_payment_intents")


async def count_pos_for_workflow(workflow_id: str) -> int:
    """Count PO rows for a specific workflow_id."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM sourcing_agent_purchase_orders WHERE idempotency_key LIKE $1",
            f"{workflow_id}:%",
        )


async def count_payments_for_workflow(workflow_id: str) -> int:
    """Count payment rows for a specific workflow_id."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM sourcing_agent_payment_intents WHERE idempotency_key LIKE $1",
            f"{workflow_id}:%",
        )
