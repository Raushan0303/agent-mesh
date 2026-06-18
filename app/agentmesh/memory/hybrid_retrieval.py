"""Hybrid retrieval — BM25 + dense vector search fused via RRF + reranker.

Combines two retrieval strategies:
  1. BM25 (Postgres full-text search) — lexical matching, good for exact keywords
  2. Dense vector search (pgvector) — semantic matching, good for paraphrases

Fuses them via Reciprocal Rank Fusion (RRF), then reranks the top-k
with a cross-encoder-style reranker (simulated with a combined
lexical+semantic score for self-contained demo).

Agent-agnostic: operates on whatever text/embeddings are stored under
a given namespace in the memory store.
"""

import logging

from app.agentmesh.memory.store import MemoryStore, get_memory_store
from app.core.config import settings

logger = logging.getLogger("agentmesh.memory.hybrid_retrieval")


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    dense_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Fuse BM25 and dense results via Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) for each result list
    where k is a tuning constant (default 60, standard value).

    This is a simple, parameter-free fusion method that doesn't require
    score calibration between the two retrieval systems.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    # BM25 ranks (1-indexed)
    for rank, result in enumerate(bm25_results, 1):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        content_map[doc_id] = result

    # Dense ranks (1-indexed)
    for rank, result in enumerate(dense_results, 1):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in content_map:
            content_map[doc_id] = result

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    return [
        {
            **content_map[doc_id],
            "rrf_score": rrf_scores[doc_id],
        }
        for doc_id in sorted_ids
    ]


def rerank(query: str, results: list[dict], top_k: int = None) -> list[dict]:
    """Cross-encoder-style reranking.

    In production, this would use a cross-encoder model (e.g., ms-marco-MiniLM)
    that takes (query, document) pairs and outputs a relevance score.

    For self-contained demo, we use a combined lexical+semantic score:
      rerank_score = semantic_similarity * 0.4 + lexical_overlap * 0.3 + rrf_score * 0.3

    The RRF score is included because it already encodes the consensus
    of both retrieval systems — a document that ranks high in both BM25
    and dense search is likely more relevant than one that ranks high
    in only one.
    """
    if top_k is None:
        top_k = settings.hybrid_retrieval_rerank_top_k

    query_tokens = set(query.lower().split())

    for result in results:
        # Semantic similarity (from dense search, if available)
        semantic = result.get("similarity", 0.0)

        # Lexical overlap (Jaccard similarity of token sets)
        doc_tokens = set(result.get("content", "").lower().split())
        if query_tokens and doc_tokens:
            overlap = len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)
        else:
            overlap = 0.0

        # RRF score (already computed by reciprocal_rank_fusion)
        rrf = result.get("rrf_score", 0.0)

        result["rerank_score"] = semantic * 0.4 + overlap * 0.3 + rrf * 0.3

    # Sort by rerank score and return top_k
    results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return results[:top_k]


async def hybrid_search(
    namespace: str,
    query: str,
    top_k: int = None,
) -> list[dict]:
    """Hybrid retrieval: BM25 + dense → RRF → reranker.

    Args:
        namespace: The agent's memory namespace
        query: The search query
        top_k: Final number of results after reranking

    Returns:
        List of {id, content, metadata, rrf_score, rerank_score} dicts
    """
    if top_k is None:
        top_k = settings.hybrid_retrieval_rerank_top_k

    fetch_k = settings.hybrid_retrieval_top_k
    store = await get_memory_store()

    # Run both searches in parallel (asyncio.gather would be ideal but
    # asyncpg connections are from a pool, so we run sequentially for safety)
    bm25_results = await store.search_bm25(namespace, query, top_k=fetch_k)
    dense_results = await store.search_dense(namespace, query, top_k=fetch_k)

    logger.info(
        "HYBRID_SEARCH namespace=%s query='%s' bm25_hits=%d dense_hits=%d",
        namespace,
        query[:50],
        len(bm25_results),
        len(dense_results),
    )

    # Fuse via RRF
    fused = reciprocal_rank_fusion(bm25_results, dense_results)

    # Rerank
    reranked = rerank(query, fused, top_k=top_k)

    logger.info(
        "HYBRID_SEARCH_DONE fused=%d reranked=%d top_score=%.4f",
        len(fused),
        len(reranked),
        reranked[0]["rerank_score"] if reranked else 0.0,
    )

    return reranked
