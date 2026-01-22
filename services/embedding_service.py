import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

load_dotenv()

class EmbeddingService:
    def __init__(self, provider: Optional[str] = None):
        """
        初始化 Embedding 服务。
        provider: 'openai' 或 'ollama'。如果不传，默认读取 .env。
        """
        self.provider = provider or os.getenv("DEFAULT_EMBEDDING_PROVIDER", "openai")
        
        if self.provider == "openai":
            self.client = OpenAIEmbeddings(
                model=os.getenv("OPENAI_EMBEDDING_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif self.provider == "ollama":
            self.client = OllamaEmbeddings(
                model=os.getenv("OLLAMA_EMBEDDING_MODEL"),
                base_url=os.getenv("OLLAMA_BASE_URL")
            )
        else:
            raise ValueError(f"不支持的 Embedding 提供商: {self.provider}")

    def get_embedding(self, text: str) -> List[float]:
        """将单段文本转化为向量"""
        try:
            return self.client.embed_query(text)
        except Exception as e:
            print(f"❌ [{self.provider}] 单段 Embedding 失败: {e}")
            return []

    def get_batch_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """批量转化文本"""
        try:
            return self.client.embed_documents(texts)
        except Exception as e:
            print(f"❌ [{self.provider}] 批量 Embedding 失败: {e}")
            return None

    def get_dimension(self) -> int:
        """获取当前模型的向量维度"""
        test_vec = self.get_vector("test")
        return len(test_vec)

# --- 快速验证 ---
if __name__ == "__main__":
    # 测试默认加载 (OpenAI)
    service = EmbeddingService()
    print(f"当前 Provider: {service.provider}")
    vec = service.get_vector("Deep-Learner 项目测试")
    print(f"维度: {len(vec)}")