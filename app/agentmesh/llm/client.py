"""LLM client adapter interface — the seam between AgentMesh and any LLM provider.

This is the platform boundary that InferRoute plugs into. Today the default
implementation calls OpenRouter/Groq/OpenAI directly via the openai SDK.
Tomorrow, swapping to InferRoute is a one-line config change (base_url).

The adapter interface is agent-agnostic — any agent's graph nodes call
`llm_client.complete(messages)` without knowing or caring who routes the call.

Provider presets:
  - openrouter: base_url=https://openrouter.ai/api/v1, models like "anthropic/claude-sonnet-4"
  - groq:       base_url=https://api.groq.com/openai/v1, models like "llama-3.3-70b-versatile"
  - openai:     base_url=https://api.openai.com/v1, models like "gpt-4o"
  - inferoute:  base_url=http://localhost:9000/v1, any model InferRoute routes to
  - custom:     any OpenAI-compatible endpoint
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger("agentmesh.llm")


# ── Message types ──


@dataclass
class LLMMessage:
    """A single chat message."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ToolCall:
    """A tool call requested by the model."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Response from an LLM completion call."""
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)  # {prompt_tokens, completion_tokens, total_tokens}
    cost_usd: float = 0.0  # from InferRoute's response (computed by InferRoute, not AgentMesh)
    raw: dict | None = None  # raw provider response for debugging
    tool_calls: list[ToolCall] = field(default_factory=list)


# ── Provider presets ──

PROVIDER_PRESETS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-3.5-sonnet",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
    },
    "inferoute": {
        "base_url": "http://localhost:9000/v1",
        "env_key": "INFERROUTE_API_KEY",
        "default_model": "auto",  # InferRoute decides
    },
}


# ── Adapter interface ──


class LLMClient(ABC):
    """Abstract LLM client — the adapter interface.

    Agent code calls `complete()` without knowing the provider.
    Swapping providers (or routing through InferRoute) is a config change,
    not a code change.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Generate a completion from a list of messages.

        Trace context (W3C traceparent) is auto-injected from the active
        OTel context — no explicit trace_id parameter needed.
        """
        ...

    # Whether complete_with_tools() is available (native function calling).
    supports_tool_calling: bool = False

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """One model turn with function calling. `messages` are OpenAI-format
        dicts (so assistant tool_calls and tool results can be replayed);
        the model either returns tool_calls or a final text answer."""
        raise NotImplementedError


# ── OpenAI-compatible implementation ──


class OpenAICompatibleClient(LLMClient):
    """Universal LLM client that works with any OpenAI-compatible API.

    Works with:
      - OpenRouter (https://openrouter.ai/api/v1)
      - Groq (https://api.groq.com/openai/v1)
      - OpenAI (https://api.openai.com/v1)
      - InferRoute (http://localhost:9000/v1) — when it's built
      - Any other OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, etc.)

    The openai SDK supports custom base_url, so one SDK handles all providers.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str = "gpt-4o",
    ):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.default_model = default_model
        logger.info(
            "LLM_CLIENT_INIT base_url=%s model=%s",
            base_url,
            default_model,
        )

    @staticmethod
    def _inject_traceparent() -> dict:
        """Inject W3C traceparent header from the current OTel context.

        This works when there's an active OTel span in the current process.
        Temporal's TracingInterceptor establishes the trace context in the
        worker process, so when the decide_node calls the LLM client,
        the trace context is active and traceparent is auto-injected.

        W3C traceparent format: 00-{trace_id(32 hex)}-{span_id(16 hex)}-{flags(2 hex)}
        """
        headers = {}
        try:
            from opentelemetry.propagate import inject
            inject(headers)
        except Exception:
            pass
        return headers

    async def complete(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Generate a completion via the OpenAI-compatible API.

        Trace context (W3C traceparent) is auto-injected from the active
        OTel context. Temporal's TracingInterceptor establishes the trace
        context in the worker, so InferRoute's spans join the same trace.
        """
        used_model = model or self.default_model

        # Convert to openai SDK format
        openai_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        # Inject W3C traceparent from the active OTel context
        extra_headers = self._inject_traceparent()

        logger.info(
            "LLM_CALL model=%s messages=%d temperature=%.1f max_tokens=%d traceparent=%s",
            used_model,
            len(openai_messages),
            temperature,
            max_tokens,
            extra_headers.get("traceparent", "none"),
        )

        # Build kwargs — some models (gpt-5, o1, o3) don't support
        # max_tokens (use max_completion_tokens) or temperature=0.
        # We try the standard params first, then retry without the
        # unsupported ones.
        kwargs = {
            "model": used_model,
            "messages": openai_messages,
            "temperature": temperature,
        }
        try:
            response = await self.client.chat.completions.create(
                **kwargs, max_tokens=max_tokens, extra_headers=extra_headers,
            )
        except Exception as e:
            err_msg = str(e)
            if "max_tokens" in err_msg and "max_completion_tokens" in err_msg:
                logger.info("LLM_RETRY with max_completion_tokens model=%s", used_model)
                try:
                    response = await self.client.chat.completions.create(
                        **kwargs, max_completion_tokens=max_tokens, extra_headers=extra_headers,
                    )
                except Exception as e2:
                    if "temperature" in str(e2):
                        logger.info("LLM_RETRY without temperature model=%s", used_model)
                        kwargs.pop("temperature", None)
                        response = await self.client.chat.completions.create(
                            **kwargs, max_completion_tokens=max_tokens, extra_headers=extra_headers,
                        )
                    else:
                        raise
            elif "temperature" in err_msg:
                logger.info("LLM_RETRY without temperature model=%s", used_model)
                kwargs.pop("temperature", None)
                try:
                    response = await self.client.chat.completions.create(
                        **kwargs, max_tokens=max_tokens, extra_headers=extra_headers,
                    )
                except Exception as e2:
                    if "max_tokens" in str(e2) and "max_completion_tokens" in str(e2):
                        response = await self.client.chat.completions.create(
                            **kwargs, max_completion_tokens=max_tokens, extra_headers=extra_headers,
                        )
                    else:
                        raise
            else:
                raise

        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        # Parse cost_usd from the response (InferRoute adds this; other providers don't)
        # The openai SDK stores extra fields in the model_extra dict
        cost_usd = 0.0
        raw_dict = response.model_dump() if hasattr(response, "model_dump") else {}
        if isinstance(raw_dict, dict):
            cost_usd = float(raw_dict.get("cost_usd", 0.0) or 0.0)

        logger.info(
            "LLM_RESPONSE model=%s content_len=%d tokens=%s cost_usd=%.6f",
            used_model,
            len(content),
            usage.get("total_tokens", "?"),
            cost_usd,
        )

        return LLMResponse(
            content=content,
            model=used_model,
            usage=usage,
            cost_usd=cost_usd,
            raw=response,
        )


    supports_tool_calling = True

    async def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        import json as _json

        used_model = model or self.default_model
        response = await self.client.chat.completions.create(
            model=used_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=self._inject_traceparent(),
        )
        msg = response.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = _json.loads(tc.function.arguments or "{}")
            except ValueError:
                args = {"_unparseable": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        raw_dict = response.model_dump() if hasattr(response, "model_dump") else {}
        return LLMResponse(
            content=msg.content or "",
            model=used_model,
            usage=usage,
            cost_usd=float((raw_dict or {}).get("cost_usd", 0.0) or 0.0),
            raw=response,
            tool_calls=calls,
        )


# ── Deterministic fallback (for tests / no API key) ──


class DeterministicLLMClient(LLMClient):
    """Fallback LLM client that produces deterministic responses.

    Used when no API key is configured (tests, CI, local dev without keys).
    Produces a reasonable rationale based on the scored suppliers, so the
    graph still works end-to-end without an LLM provider.
    """

    async def complete(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        # Find the user message containing supplier data
        user_content = ""
        for m in messages:
            if m.role == "user":
                user_content = m.content

        # Generate a deterministic rationale
        rationale = self._generate_rationale(user_content)

        return LLMResponse(
            content=rationale,
            model="deterministic-fallback",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    def _generate_rationale(self, context: str) -> str:
        """Generate a deterministic rationale from the context string."""
        # Extract supplier info from the context
        lines = []
        lines.append("Based on the scored suppliers, I recommend selecting the top-ranked supplier.")
        lines.append("")
        lines.append("Key factors considered:")
        lines.append("- Price: lower is better (weighted 40%)")
        lines.append("- Rating: higher is better (weighted 30%)")
        lines.append("- On-time rate: higher is better (weighted 20%)")
        lines.append("- Lead time: shorter is better (weighted 10%)")
        lines.append("")
        lines.append("The selected supplier has the best combined score across these factors.")
        if "previously sourced" in context.lower() or "positive feedback" in context.lower():
            lines.append("")
            lines.append("Additionally, we have previously sourced this item with positive feedback, "
                         "which increases confidence in the selection.")

        return "\n".join(lines)


# ── Factory ──


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get or create the LLM client singleton.

    Resolution order:
      1. If AGENTMESH_LLM_PROVIDER is set, use that provider's preset
      2. If the provider's API key is in env vars, create OpenAICompatibleClient
      3. If no API key, fall back to DeterministicLLMClient (tests/CI)

    This means:
      - Set OPENROUTER_API_KEY in .env → uses OpenRouter
      - Set GROQ_API_KEY in .env → uses Groq
      - Set INFERROUTE_API_KEY in .env → uses InferRoute (when built)
      - No key set → deterministic fallback (tests pass without API keys)
    """
    global _client
    if _client is not None:
        return _client

    provider = settings.llm_provider

    if provider == "deterministic":
        logger.info("LLM_CLIENT=deterministic (explicit config)")
        _client = DeterministicLLMClient()
        return _client

    # Look up provider preset
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        # Custom base_url from settings
        base_url = settings.llm_base_url
        api_key = settings.llm_api_key or os.environ.get("LLM_API_KEY", "")
        model = settings.llm_model or "gpt-4o"
        if not api_key:
            logger.warning(
                "LLM_CLIENT=deterministic (no API key for custom provider %s)",
                provider,
            )
            _client = DeterministicLLMClient()
            return _client
        _client = OpenAICompatibleClient(base_url, api_key, model)
        return _client

    # Use preset — check settings.llm_api_key first, then env var
    api_key = settings.llm_api_key or os.environ.get(preset["env_key"], "")
    if not api_key:
        logger.warning(
            "LLM_CLIENT=deterministic (no %s env var for provider %s)",
            preset["env_key"],
            provider,
        )
        _client = DeterministicLLMClient()
        return _client

    # Prefer settings.llm_base_url over preset (so .env can override the endpoint)
    base_url = settings.llm_base_url or preset["base_url"]
    model = settings.llm_model or preset["default_model"]
    _client = OpenAICompatibleClient(base_url, api_key, model)
    return _client


def reset_llm_client() -> None:
    """Reset the singleton (for tests)."""
    global _client
    _client = None
