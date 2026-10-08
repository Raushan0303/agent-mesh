"""LLM layer — universal LLM client adapter.

This is the seam between AgentMesh and any LLM provider.
Today: OpenRouter, Groq, OpenAI (via openai SDK with custom base_url).
Tomorrow: InferRoute (just another base_url).

Agent code calls `llm_client.complete(messages)` without knowing
or caring who routes the call.
"""

from app.agentmesh.llm.client import (
    DeterministicLLMClient,
    LLMClient,
    LLMMessage,
    LLMResponse,
    OpenAICompatibleClient,
    PROVIDER_PRESETS,
    ToolCall,
    get_llm_client,
    reset_llm_client,
)

__all__ = [
    "DeterministicLLMClient",
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "OpenAICompatibleClient",
    "PROVIDER_PRESETS",
    "ToolCall",
    "get_llm_client",
    "reset_llm_client",
]
