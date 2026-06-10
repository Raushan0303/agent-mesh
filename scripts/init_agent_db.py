"""One-time script: create the 'agentmesh' database and idempotency tables.

Usage:
    source venv/bin/activate
    python scripts/init_agent_db.py
"""

import asyncio

import asyncpg

from app.core.config import settings
from app.agents.sourcing_agent.db import init_tables, close_db_pool


async def create_database_if_not_exists():
    """Create the 'agentmesh' database if it doesn't exist."""
    # Connect to the default 'postgres' database to check/create
    admin_dsn = settings.postgres_dsn.replace("/temporal", "/temporal")
    conn = await asyncpg.connect(admin_dsn)
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = 'agentmesh'"
        )
        if not exists:
            await conn.execute("CREATE DATABASE agentmesh")
            print("Created database: agentmesh")
        else:
            print("Database 'agentmesh' already exists")
    finally:
        await conn.close()


async def main():
    print("Initializing AgentMesh database...")
    await create_database_if_not_exists()
    await init_tables()
    print("Tables created: sourcing_agent_purchase_orders, sourcing_agent_payment_intents")
    await close_db_pool()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
