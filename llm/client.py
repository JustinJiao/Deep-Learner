# llm/client.py
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
try:
    from langchain_anthropic import ChatAnthropic
except Exception:  # pragma: no cover
    ChatAnthropic = None
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import AppConfig


class LLMService:
    """
    聊天模型统一封装（确定性输出）
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or AppConfig.LLM_PROVIDER
        self._client_cache: dict[tuple[str, str], object] = {}

    def _default_model_for_provider(self, provider: str) -> str:
        if provider == "openai":
            return AppConfig.OPENAI_CHAT_MODEL
        if provider == "ollama":
            return AppConfig.OLLAMA_CHAT_MODEL
        if provider == "anthropic":
            return AppConfig.ANTHROPIC_COMPOSE_MODEL
        raise ValueError(f"不支持的 LLM Provider: {provider}")

    def _resolve_route(self, task: str) -> tuple[str, str]:
        task_norm = str(task or "default").strip().lower()
        if AppConfig.LLM_ROUTING_ENABLED:
            if task_norm == "compose":
                return ("openai", AppConfig.OPENAI_COMPOSE_MODEL)
            if task_norm in {"verify", "rewrite"}:
                model = (
                    AppConfig.OPENAI_VERIFY_MODEL
                    if task_norm == "verify"
                    else AppConfig.OPENAI_REWRITE_MODEL
                )
                return ("openai", model)
            if task_norm == "memory":
                return ("anthropic", AppConfig.ANTHROPIC_MEMORY_MODEL)

        provider = self.provider
        return (provider, self._default_model_for_provider(provider))

    def _build_client(self, provider: str, model: str):
        if provider == "openai":
            return ChatOpenAI(
                model=model,
                api_key=AppConfig.OPENAI_API_KEY,
                temperature=AppConfig.OPENAI_TEMPERATURE,
                timeout=AppConfig.OPENAI_TIMEOUT_SECONDS,
                max_retries=AppConfig.OPENAI_MAX_RETRIES,
            )
        if provider == "ollama":
            return ChatOllama(
                model=model,
                base_url=AppConfig.OLLAMA_BASE_URL,
                temperature=AppConfig.OLLAMA_TEMPERATURE,
                timeout=AppConfig.OLLAMA_TIMEOUT_SECONDS,
            )
        if provider == "anthropic":
            if ChatAnthropic is None:
                raise ImportError(
                    "langchain-anthropic is not installed; Anthropic routing is unavailable"
                )
            return ChatAnthropic(
                model=model,
                api_key=AppConfig.ANTHROPIC_API_KEY,
                temperature=AppConfig.ANTHROPIC_TEMPERATURE,
                timeout=AppConfig.ANTHROPIC_TIMEOUT_SECONDS,
                max_retries=AppConfig.ANTHROPIC_MAX_RETRIES,
            )
        raise ValueError(f"不支持的 LLM Provider: {provider}")

    def _get_client(self, provider: str, model: str):
        key = (provider, model)
        client = self._client_cache.get(key)
        if client is None:
            client = self._build_client(provider, model)
            self._client_cache[key] = client
        return client

    def chat_completion(
        self,
        prompt: str,
        system_prompt: str,
        task: str = "default",
    ) -> str:
        provider, model = self._resolve_route(task)
        llm = self._get_client(provider, model)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        return response.content
