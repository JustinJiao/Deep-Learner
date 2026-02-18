# llm/embeddings.py
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from config.settings import AppConfig


class EmbeddingService:
    """
    向量模型统一封装
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or AppConfig.EMBEDDING_PROVIDER

        if self.provider == "openai":
            self.client = OpenAIEmbeddings(
                model=AppConfig.OPENAI_EMBED_MODEL,
                api_key=AppConfig.OPENAI_API_KEY,
            )
        elif self.provider == "ollama":
            self.client = OllamaEmbeddings(
                model=AppConfig.OLLAMA_EMBED_MODEL,
                base_url=AppConfig.OLLAMA_BASE_URL,
            )
        else:
            raise ValueError(f"不支持的 Embedding Provider: {self.provider}")

    def embed_query(self, text: str) -> List[float]:
        return self.client.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed_documents(texts)
