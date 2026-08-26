import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ so non-prefixed keys (OPENAI_API_KEY, etc.)
# are available to code that reads os.environ directly.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AGENTMESH_",
        extra="ignore",
    )

    temporal_host: str = "localhost:7233"
    temporal_namespace: str = "default"

    postgres_dsn: str = "postgresql://temporal:temporal@localhost:5434/temporal"

    # AgentMesh app database (same Postgres instance, separate database)
    agent_db_dsn: str = "postgresql://temporal:temporal@localhost:5434/agentmesh"

    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8000

    max_concurrent_activities: int = 100

    # ── Week 6: RAG / memory / observability ──

    # Embedding dimension — 384 for all-MiniLM-L6-v2, 256 for hash fallback
    embedding_dim: int = 384

    # Semantic cache
    semantic_cache_ttl_seconds: int = 3600  # 1 hour
    semantic_cache_similarity_threshold: float = 0.95

    # Hybrid retrieval
    hybrid_retrieval_top_k: int = 10
    hybrid_retrieval_rerank_top_k: int = 5

    # OpenTelemetry / Jaeger
    otel_endpoint: str = "http://localhost:4318/v1/traces"
    otel_service_name: str = "agentmesh"
    otel_enabled: bool = True

    # Eval thresholds (CI gate)
    eval_min_tool_accuracy: float = 0.90
    eval_min_completion_rate: float = 0.95
    eval_min_rag_recall: float = 0.80

    # ── LLM client ──
    # Provider: "openrouter", "groq", "openai", "inferoute", "custom", "deterministic"
    # When set to a provider, the client looks up the corresponding API key env var.
    # When set to "deterministic", uses a fallback that needs no API key (tests/CI).
    llm_provider: str = "openrouter"
    llm_model: str = ""  # empty = use provider's default model
    llm_base_url: str = ""  # empty = use provider preset; set for "custom"
    llm_api_key: str = ""  # empty = read from env var (OPENROUTER_API_KEY, GROQ_API_KEY, etc.)

    # ── Week 12-13: Delivery + Sandbox + Cost + Verification ──

    # Redis (SSE event bus + event history for reconnection)
    redis_url: str = "redis://localhost:6379"

    # SSE streaming
    sse_buffer_size: int = 100  # max events buffered before closing slow client
    sse_history_ttl_seconds: int = 3600  # 1 hour — how long event history is kept for reconnection

    # Cost ceilings
    default_cost_budget_usd: float = 0.50  # default per-workflow budget

    # Webhook delivery
    webhook_timeout_seconds: float = 10.0
    webhook_retry_attempts: int = 3
    webhook_hmac_secret: str = ""  # empty = generate per-request (returns in start response)


settings = Settings()
