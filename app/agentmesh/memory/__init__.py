"""Memory layer — pgvector store, hybrid retrieval, semantic cache.

This is the third kind of state in the system:
  1. Temporal Event History — durability, per-execution
  2. LangGraph Checkpoints — resumable graph state, per-execution
  3. Agent Memory (this) — cross-session, not resumable-execution-state
"""

from app.agentmesh.memory.store import (
    MemoryEntry,
    MemoryStore,
    embed_text,
    get_memory_store,
    close_memory_store,
)
from app.agentmesh.memory.hybrid_retrieval import (
    hybrid_search,
    reciprocal_rank_fusion,
    rerank,
)
from app.agentmesh.memory.semantic_cache import (
    SemanticCache,
    get_semantic_cache,
    close_semantic_cache,
)

__all__ = [
    "MemoryEntry",
    "MemoryStore",
    "embed_text",
    "get_memory_store",
    "close_memory_store",
    "hybrid_search",
    "reciprocal_rank_fusion",
    "rerank",
    "SemanticCache",
    "get_semantic_cache",
    "close_semantic_cache",
]
