"""Agent long-term memory in pgvector — cross-session memory store.

This is deliberately kept separate from:
- LangGraph's checkpointer (short-term, resumable, per-execution state)
- Temporal's Event History (durability, per-execution event log)

This is a different kind of state: cross-session memory. When a founder
asks "have we sourced USB-C cables before?", the Decide node queries
this store for past sourcing decisions — not the checkpointer (which
only has the current execution's state) or Temporal's Event History
(which only has one workflow's events).

Uses pgvector for dense vector similarity search. The store is
namespaced per agent so memories never collide across agents.

Embeddings: uses sentence-transformers/all-MiniLM-L6-v2 (384-dim) — a
real semantic embedding model from HuggingFace. Falls back to a
deterministic hash-based embedding if the model isn't available (e.g.,
in CI without the model downloaded).
"""

import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field

import asyncpg

from app.core.config import settings

logger = logging.getLogger("agentmesh.memory.store")


# ── Embedding model (lazy-loaded singleton) ──

_embedding_model = None
_embedding_dim_actual = None


def _get_embedding_model():
    """Lazy-load the sentence-transformers model.

    Returns None if the model isn't available (falls back to hash embeddings).
    """
    global _embedding_model, _embedding_dim_actual
    if _embedding_model is None and _embedding_dim_actual is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            _embedding_dim_actual = 384
            logger.info("EMBEDDING_MODEL_LOADED model=all-MiniLM-L6-v2 dim=%d", _embedding_dim_actual)
        except Exception as e:
            logger.warning("EMBEDDING_MODEL_FALLBACK hash-based (reason: %s)", e)
            _embedding_dim_actual = settings.embedding_dim  # fallback to hash
    return _embedding_model


def embed_text(text: str, dim: int = None) -> list[float]:
    """Embed text into a fixed-dimensional vector.

    Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim) for real semantic
    embeddings. Falls back to a deterministic hash-based embedding if the
    model isn't available (e.g., in CI without the model downloaded).

    The vector is normalized to unit length so cosine similarity works.
    """
    model = _get_embedding_model()

    if model is not None:
        # Real semantic embedding
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    # Fallback: deterministic hash-based embedding
    if dim is None:
        dim = settings.embedding_dim

    vector = [0.0] * dim
    tokens = text.lower().split()

    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        idx = h % dim
        val = ((h >> 8) % 1000) / 1000.0 - 0.5
        vector[idx] += val

    padded = f" {text.lower()} "
    for i in range(len(padded) - 2):
        trigram = padded[i : i + 3]
        h = int(hashlib.sha256(trigram.encode()).hexdigest(), 16)
        idx = h % dim
        vector[idx] += 0.3

    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]

    return vector


# ── Memory entry ──


@dataclass
class MemoryEntry:
    """A single memory entry in the store."""
    id: str
    namespace: str
    content: str
    metadata: dict
    embedding: list[float] = field(default_factory=list)
    created_at: float = 0.0


# ── Memory store ──


class MemoryStore:
    """Namespaced cross-session memory store backed by pgvector.

    The store is agent-agnostic — it operates on namespaces and text.
    The agent decides what to store and what to query.
    """

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def setup(self) -> None:
        """Create the memory table and vector index if they don't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_memory (
                    id          TEXT PRIMARY KEY,
                    namespace   TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    metadata    JSONB NOT NULL DEFAULT '{}',
                    embedding   vector(%s),
                    created_at  DOUBLE PRECISION NOT NULL,
                    tsv         tsvector
                )
            """ % settings.embedding_dim)

            # HNSW index for fast vector similarity search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_memory_embedding
                ON agent_memory USING hnsw (embedding vector_cosine_ops)
            """)

            # GIN index for full-text search (BM25 proxy)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_memory_tsv
                ON agent_memory USING gin(tsv)
            """)

            # Index on namespace for filtering
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_memory_namespace
                ON agent_memory(namespace)
            """)

            logger.info("MEMORY_STORE_INITIALIZED dim=%d", settings.embedding_dim)

    async def store(
        self,
        namespace: str,
        content: str,
        metadata: dict | None = None,
    ) -> str:
        """Store a memory entry under a namespace.

        Args:
            namespace: The agent's namespace (e.g., "sourcing-agent")
            content: The text content to store
            metadata: Optional metadata dict (stored as JSONB)

        Returns:
            The generated entry ID
        """
        entry_id = str(uuid.uuid4())
        embedding = embed_text(content)
        meta = metadata or {}
        now = time.time()

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_memory (id, namespace, content, metadata, embedding, created_at, tsv)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6,
                        to_tsvector('english', $3))
                """,
                entry_id,
                namespace,
                content,
                json.dumps(meta),
                str(embedding),
                now,
            )

        logger.info(
            "MEMORY_STORED id=%s namespace=%s content_len=%d",
            entry_id,
            namespace,
            len(content),
        )
        return entry_id

    async def search_dense(
        self,
        namespace: str,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """Dense vector similarity search (cosine similarity via pgvector).

        Returns list of {id, content, metadata, similarity} dicts.
        """
        query_embedding = embed_text(query)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, metadata,
                       1 - (embedding <=> $1::vector) as similarity
                FROM agent_memory
                WHERE namespace = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                str(query_embedding),
                namespace,
                top_k,
            )

        return [
            {
                "id": r["id"],
                "content": r["content"],
                "metadata": r["metadata"],
                "similarity": float(r["similarity"]),
            }
            for r in rows
        ]

    async def search_bm25(
        self,
        namespace: str,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """BM25-style full-text search via Postgres ts_rank.

        Returns list of {id, content, metadata, rank} dicts.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, content, metadata,
                       ts_rank(tsv, plainto_tsquery('english', $1)) as rank
                FROM agent_memory
                WHERE namespace = $2 AND tsv @@ plainto_tsquery('english', $1)
                ORDER BY rank DESC
                LIMIT $3
                """,
                query,
                namespace,
                top_k,
            )

        return [
            {
                "id": r["id"],
                "content": r["content"],
                "metadata": r["metadata"],
                "rank": float(r["rank"]),
            }
            for r in rows
        ]

    async def get_all(self, namespace: str, limit: int = 100) -> list[dict]:
        """Get all entries in a namespace (for testing/debugging)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, content, metadata, created_at FROM agent_memory WHERE namespace = $1 LIMIT $2",
                namespace,
                limit,
            )
        return [dict(r) for r in rows]

    async def clear(self, namespace: str) -> int:
        """Clear all entries in a namespace. Returns count deleted."""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT count(*) FROM agent_memory WHERE namespace = $1",
                namespace,
            )
            await conn.execute(
                "DELETE FROM agent_memory WHERE namespace = $1",
                namespace,
            )
        return count or 0


# ── Singleton ──

_store: MemoryStore | None = None
_pool: asyncpg.Pool | None = None


async def get_memory_store() -> MemoryStore:
    """Get or create the memory store singleton."""
    global _store, _pool
    if _store is None:
        _pool = await asyncpg.create_pool(
            settings.agent_db_dsn,
            min_size=2,
            max_size=10,
        )
        _store = MemoryStore(_pool)
        await _store.setup()
    return _store


async def close_memory_store() -> None:
    """Close the memory store pool (for tests)."""
    global _store, _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        _store = None


async def get_memory_store_with_pool(pool: asyncpg.Pool) -> MemoryStore:
    """Create a memory store using an existing pool (for tests)."""
    store = MemoryStore(pool)
    await store.setup()
    return store
