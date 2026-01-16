import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    def __init__(self):
        # 使用 2026 年主流的 text-embedding-3-small 模型
        # 该模型平衡了成本与性能，输出维度为 1536
        self.client = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def get_vector(self, text: str):
        """将单段文本转化为向量"""
        return self.client.embed_query(text)

    def get_batch_vectors(self, texts: list):
        """
        批量转化文本（推荐在 DualWriter 中使用）
        优点：减少网络往返耗时，提高 Ingestion 效率
        """
        try:
            vectors = self.client.embed_documents(texts)
            return vectors
        except Exception as e:
            print(f"❌ Embedding 失败: {e}")
            return None

# --- 简单测试脚本 ---
if __name__ == "__main__":
    service = EmbeddingService()
    test_text = "Spark 统一内存管理机制包括堆内和堆外内存。"
    vector = service.get_vector(test_text)
    print(f"向量长度: {len(vector)}") # 应该是 1536
    print(f"向量前五位: {vector[:5]}")