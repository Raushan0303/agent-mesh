"""RAG gateway endpoints — knowledge base ingestion and retrieval.

These endpoints expose agent-mesh's RAG layer to the UI and external
callers. The retrieval engine (embeddings, BM25, dense search, RRF
fusion, reranking) lives in app/agentmesh/memory/. This router wraps
it in HTTP so the UI can:

  1. Upload documents → chunk → embed → store in pgvector  (/rag/upsert)
  2. Query the knowledge base → hybrid search → ranked chunks   (/rag/query)

Agent-agnostic: operates on namespaces. Each agent (or knowledge base)
gets its own namespace so documents never collide.
"""

import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.agentmesh.memory.chunking import chunk_text
from app.agentmesh.memory.hybrid_retrieval import hybrid_search
from app.agentmesh.memory.store import get_memory_store

logger = logging.getLogger("agentmesh.gateway.rag")

router = APIRouter(prefix="/rag", tags=["rag"])


# ── Request / response models ──


class UpsertRequest(BaseModel):
    """Upload documents into a knowledge base namespace.

    Documents are chunked, embedded, and stored in pgvector.
    """
    namespace: str = Field(..., description="Knowledge base namespace (e.g., 'sourcing-agent', 'hiring-policies')")
    documents: list[dict] = Field(
        ...,
        description="List of {content: str, metadata?: dict}. Each document is chunked before storage.",
    )
    chunk_size: int = Field(512, description="Target chunk size in tokens (word-level approximation)")
    overlap: int = Field(50, description="Token overlap between adjacent chunks")
    chunk_strategy: str = Field("fixed", description="'fixed' or 'sentence'")


class QueryRequest(BaseModel):
    """Query a knowledge base namespace via hybrid retrieval."""
    namespace: str = Field(..., description="Knowledge base namespace to search")
    query: str = Field(..., description="The search query")
    top_k: int = Field(5, description="Number of results to return after reranking")


# ── Endpoints ──


@router.post("/upsert")
async def upsert_documents(request: Request) -> JSONResponse:
    """Ingest documents into a knowledge base.

    Pipeline: documents → chunk → embed → store in pgvector.

    Each document is split into overlapping chunks. Each chunk is
    embedded and stored as a separate row in the agent_memory table
    under the given namespace. The original document ID and chunk index
    are stored in metadata for traceability.

    Request body:
        {
            "namespace": "sourcing-agent",
            "documents": [
                {"content": "Supplier A: USB-C cables, $2.50/unit, 2-week lead time...", "metadata": {"source": "catalog.pdf"}},
                {"content": "Supplier B: HDMI cables, $3.10/unit, 1-week lead time...", "metadata": {"source": "catalog.pdf"}}
            ],
            "chunk_size": 512,
            "overlap": 50,
            "chunk_strategy": "fixed"
        }

    Returns:
        {
            "namespace": "sourcing-agent",
            "documents_uploaded": 2,
            "chunks_stored": 8,
            "chunks_per_doc": [3, 5]
        }
    """
    body = await request.json()
    try:
        req = UpsertRequest(**body)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    store = await get_memory_store()

    total_chunks = 0
    chunks_per_doc = []

    for doc in req.documents:
        content = doc.get("content", "")
        if not content.strip():
            chunks_per_doc.append(0)
            continue

        doc_metadata = doc.get("metadata", {})
        doc_id = doc_metadata.get("doc_id", str(uuid.uuid4()))

        chunks = chunk_text(
            content,
            chunk_size=req.chunk_size,
            overlap=req.overlap,
            strategy=req.chunk_strategy,
        )

        for idx, chunk in enumerate(chunks):
            chunk_metadata = {
                **doc_metadata,
                "doc_id": doc_id,
                "chunk_index": idx,
                "total_chunks": len(chunks),
            }
            await store.store(
                namespace=req.namespace,
                content=chunk,
                metadata=chunk_metadata,
            )

        total_chunks += len(chunks)
        chunks_per_doc.append(len(chunks))

    logger.info(
        "RAG_UPSERT namespace=%s documents=%d chunks_stored=%d",
        req.namespace, len(req.documents), total_chunks,
    )

    return JSONResponse({
        "namespace": req.namespace,
        "documents_uploaded": len(req.documents),
        "chunks_stored": total_chunks,
        "chunks_per_doc": chunks_per_doc,
    })


@router.post("/query")
async def query_knowledge_base(request: Request) -> JSONResponse:
    """Query a knowledge base via hybrid retrieval (BM25 + dense + RRF + rerank).

    Request body:
        {
            "namespace": "sourcing-agent",
            "query": "USB-C cable suppliers under $3",
            "top_k": 5
        }

    Returns:
        {
            "namespace": "sourcing-agent",
            "query": "USB-C cable suppliers under $3",
            "results": [
                {
                    "id": "abc-123",
                    "content": "Supplier A: USB-C cables, $2.50/unit...",
                    "metadata": {"doc_id": "...", "chunk_index": 0},
                    "rrf_score": 0.0312,
                    "rerank_score": 0.8421
                }
            ],
            "count": 1
        }
    """
    body = await request.json()
    try:
        req = QueryRequest(**body)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    results = await hybrid_search(
        namespace=req.namespace,
        query=req.query,
        top_k=req.top_k,
    )

    logger.info(
        "RAG_QUERY namespace=%s query='%s' results=%d",
        req.namespace, req.query[:50], len(results),
    )

    return JSONResponse({
        "namespace": req.namespace,
        "query": req.query,
        "results": results,
        "count": len(results),
    })


@router.get("/namespaces/{namespace}/stats")
async def get_namespace_stats(namespace: str) -> JSONResponse:
    """Get document/chunk count for a knowledge base namespace."""
    store = await get_memory_store()
    entries = await store.get_all(namespace, limit=10000)

    # Group by doc_id to count unique documents
    doc_ids = set()
    for entry in entries:
        meta = entry.get("metadata", {})
        if isinstance(meta, dict):
            doc_ids.add(meta.get("doc_id", entry["id"]))

    return JSONResponse({
        "namespace": namespace,
        "total_chunks": len(entries),
        "total_documents": len(doc_ids),
    })


@router.delete("/namespaces/{namespace}")
async def clear_namespace(namespace: str) -> JSONResponse:
    """Clear all documents from a knowledge base namespace."""
    store = await get_memory_store()
    deleted = await store.clear(namespace)

    logger.info("RAG_CLEAR namespace=%s deleted=%d", namespace, deleted)

    return JSONResponse({
        "namespace": namespace,
        "deleted": deleted,
    })
