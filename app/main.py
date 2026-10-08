from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agentmesh.gateway.routes import router as gateway_router
from app.agentmesh.gateway.trace_routes import router as trace_router
from app.agentmesh.gateway.benchmark_routes import router as benchmark_router
from app.agentmesh.gateway.rag_routes import router as rag_router
from app.agentmesh.observability.tracing import setup_tracing
from app.agentmesh.mcp_server import build_mcp_server
import app.agents.sourcing_agent  # noqa: F401 — triggers self-registration into AGENT_REGISTRY
import app.agents.hiring_agent  # noqa: F401 — second agent, same engine, harder problem
import app.agents.benchmark_agent  # noqa: F401 — benchmark agent for load testing

# Initialize OpenTelemetry tracing on startup — exports to Jaeger via OTLP
setup_tracing()

# Real MCP server (JSON-RPC tools/list + tools/call over Streamable HTTP)
# for the Tool Registry, served at POST /mcp.
mcp_server = build_mcp_server()
mcp_app = mcp_server.streamable_http_app(streamable_http_path="/mcp")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # A mounted Starlette app's own lifespan does not run, so start the MCP
    # session manager here.
    async with mcp_server.session_manager.run():
        yield


app = FastAPI(
    title="AgentMesh",
    description="Durable agent execution platform — Temporal + LangGraph hybrid architecture",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gateway_router)
app.include_router(trace_router)
app.include_router(benchmark_router)
app.include_router(rag_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agentmesh"}


# Mounted last so every route defined above (including /health) takes precedence; /mcp is the
# only path the MCP app serves.
app.mount("/", mcp_app)
