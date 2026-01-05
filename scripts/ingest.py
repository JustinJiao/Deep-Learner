import os
from pymilvus import MilvusClient
from services.llm_service import embedding_model, get_actual_dim  # 复用你的 Ollama Embedding

DIMENSION = get_actual_dim()
print(f"--- 检测到 nomic 模型真实维度为: {DIMENSION} ---")

# 1. 配置
COLLECTION_NAME = "deep_learner_docs"
DB_FILE = "./database/milvus_demo.db"  # 使用 Milvus Lite 存储在本地

client = MilvusClient(DB_FILE)

if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)
    print("已删除旧的维度集合")

# 2. 创建集合 (Schema)
client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=DIMENSION,  # nomic-embed-text 的维度通常是 4096 (Llama/Qwen 风格)
)

def ingest_data(file_path):
    print(f"--- 正在处理文件: {file_path} ---")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # V1 简单切分：按段落切分
    chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 10]
    
    data = []
    for i, chunk in enumerate(chunks):
        # 调用 Ollama 生成向量
        embedding = embedding_model.embed_query(chunk)
        data.append({"id": i, "vector": embedding, "text": chunk})
        print(f"已处理第 {i+1} 个数据块")
    
    # 插入 Milvus
    client.insert(collection_name=COLLECTION_NAME, data=data)
    print(f"--- 成功存入 {len(data)} 条数据至 Milvus ---")

if __name__ == "__main__":
    # 确保 data 目录下有文本文件
    ingest_data("./data/knowledge.txt")