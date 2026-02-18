# llm/client.py
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import AppConfig


class LLMService:
    """
    聊天模型统一封装（确定性输出）
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or AppConfig.LLM_PROVIDER

        if self.provider == "openai":
            self.llm = ChatOpenAI(
                model=AppConfig.OPENAI_CHAT_MODEL,
                api_key=AppConfig.OPENAI_API_KEY,
                temperature=0,
            )
        elif self.provider == "ollama":
            self.llm = ChatOllama(
                model=AppConfig.OLLAMA_CHAT_MODEL,
                base_url=AppConfig.OLLAMA_BASE_URL,
                temperature=0,
                timeout=60,
            )
        else:
            raise ValueError(f"不支持的 LLM Provider: {self.provider}")

    def chat_completion(
        self,
        prompt: str,
        system_prompt: str,
    ) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content
