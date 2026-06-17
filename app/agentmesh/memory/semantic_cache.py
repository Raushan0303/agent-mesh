"""Semantic cache — cache LLM responses keyed by embedding similarity.

Caches responses keyed by embedding similarity of the prompt. When a
near-duplicate prompt arrives (e.g., "steel pipes, 500 units" vs "steel
pipes, 500 pieces"), the cache returns the stored response if the
similarity exceeds the threshold.

Namespaced per agent so caches never collide across agents.

TTL/staleness policy: entries expire after semantic_cache_ttl_seconds.
The cache also supports explicit invalidation.
"""

import logging
import time

import asyncpg

from app.agentmesh.memory.store import embed_text
from app.core.config import settings

logger = logging.getLogger("agentmesh.memory.semantic_cache")


class SemanticCache:
    """Semantic cache backed by pgvector.

    Stores (prompt_embedding, response, namespace, timestamp) tuples.
    On lookup, finds the nearest prompt by cosine similarity and returns
    the cached response if similarity >= threshold and entry is not expired.
    """

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def setup(self) -> None:
        """Create the cache table if it doesn't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id          TEXT PRIMARY KEY,
                    namespace   TEXT NOT NULL,
                    prompt      TEXT NOT NULL,
                    response    TEXT NOT NULL,
                    embedding   vector(%s),
                    created_at  DOUBLE PRECISION NOT NULL,
                    ttl_seconds INTEGER NOT NULL DEFAULT 3600
                )
            """ % settings.embedding_dim)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_cache_embedding
                ON semantic_cache USING hnsw (embedding vector_cosine_ops)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_semantic_cache_namespace
                ON semantic_cache(namespace)
            """)

            logger.info("SEMANTIC_CACHE_INITIALIZED dim=%d", settings.embedding_dim)

    async def get(self, namespace: str, prompt: str) -> dict | None:
        """Look up a cached response by semantic similarity.

        Returns {response, similarity, age_seconds} if cache hit, None if miss.
        """
        query_embedding = embed_text(prompt)
        now = time.time()

        async with self.pool.acquire() as conn:
            # Find the nearest cached prompt
            row = await conn.fetchrow(
                """
                SELECT id, response, created_at, ttl_seconds,
                       1 - (embedding <=> $1::vector) as similarity
                FROM semantic_cache
                WHERE namespace = $2
                ORDER BY embedding <=> $1::vector
                LIMIT 1
                """,
                str(query_embedding),
                namespace,
            )

        if row is None:
            logger.debug("SEMANTIC_CACHE_MISS namespace=%s (empty cache)", namespace)
            return None

        similarity = float(row["similarity"])
        age = now - float(row["created_at"])
        ttl = int(row["ttl_seconds"])

        # Check similarity threshold
        if similarity < settings.semantic_cache_similarity_threshold:
            logger.debug(
                "SEMANTIC_CACHE_MISS namespace=%s similarity=%.4f threshold=%.4f",
                namespace,
                similarity,
                settings.semantic_cache_similarity_threshold,
            )
            return None

        # Check TTL
        if age > ttl:
            logger.debug(
                "SEMANTIC_CACHE_MISS namespace=%s age=%.1fs ttl=%ds (expired)",
                namespace,
                age,
                ttl,
            )
            return None

        logger.info(
            "SEMANTIC_CACHE_HIT namespace=%s similarity=%.4f age=%.1fs",
            namespace,
            similarity,
            age,
        )

        return {
            "response": row["response"],
            "similarity": similarity,
            "age_seconds": age,
        }

    async def put(
        self,
        namespace: str,
        prompt: str,
        response: str,
        ttl_seconds: int = None,
    ) -> str:
        """Store a response in the cache.

        Args:
            namespace: The agent's cache namespace
            prompt: The prompt text
            response: The response to cache
            ttl_seconds: TTL in seconds (default: settings.semantic_cache_ttl_seconds)

        Returns:
            The cache entry ID
        """
        import uuid

        entry_id = str(uuid.uuid4())
        embedding = embed_text(prompt)
        now = time.time()
        ttl = ttl_seconds or settings.semantic_cache_ttl_seconds

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO semantic_cache (id, namespace, prompt, response, embedding, created_at, ttl_seconds)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                entry_id,
                namespace,
                prompt,
                response,
                str(embedding),
                now,
                ttl,
            )

        logger.info(
            "SEMANTIC_CACHE_PUT namespace=%s prompt_len=%d response_len=%d ttl=%ds",
            namespace,
            len(prompt),
            len(response),
            ttl,
        )
        return entry_id

    async def clear(self, namespace: str) -> int:
        """Clear all cache entries for a namespace. Returns count deleted."""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                "DELETE FROM semantic_cache WHERE namespace = $1",
                namespace,
            )
        return count or 0

    async def stats(self, namespace: str = None) -> dict:
        """Get cache statistics."""
        async with self.pool.acquire() as conn:
            if namespace:
                total = await conn.fetchval(
                    "SELECT count(*) FROM semantic_cache WHERE namespace = $1",
                    namespace,
                )
            else:
                total = await conn.fetchval("SELECT count(*) FROM semantic_cache")

        return {"total_entries": total or 0}


# ── Singleton ──

_cache: SemanticCache | None = None
_pool: asyncpg.Pool | None = None


async def get_semantic_cache() -> SemanticCache:
    """Get or create the semantic cache singleton."""
    global _cache, _pool
    if _cache is None:
        _pool = await asyncpg.create_pool(
            settings.agent_db_dsn,
            min_size=2,
            max_size=10,
        )
        _cache = SemanticCache(_pool)
        await _cache.setup()
    return _cache


async def close_semantic_cache() -> None:
    """Close the semantic cache pool (for tests)."""
    global _cache, _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        _cache = None
