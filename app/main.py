from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agentmesh.gateway.routes import router as gateway_router
from app.agentmesh.gateway.rag_routes import router as rag_router
from app.agentmesh.observability.tracing import setup_tracing
import app.agents.sourcing_agent  # noqa: F401 — triggers self-registration into AGENT_REGISTRY
import app.agents.hiring_agent  # noqa: F401 — second agent, same engine, harder problem
import app.agents.benchmark_agent  # noqa: F401 — benchmark agent for load testing

# Initialize OpenTelemetry tracing on startup — exports to Jaeger via OTLP
setup_tracing()

app = FastAPI(
    title="AgentMesh",
    description="Durable agent execution platform — Temporal + LangGraph hybrid architecture",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gateway_router)
app.include_router(rag_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agentmesh"}
