from pymilvus import MilvusClient
from services.llm_service import embedding_model

# 数据库配置
COLLECTION_NAME = "deep_learner_docs"
client = MilvusClient("./database/milvus_demo.db")

def retriever_node(state):
    search_query = state['plan']['search_query']
    print(f"--- [Retriever] 正在检索 Milvus: {search_query} ---")
    
    # 将关键词向量化
    query_vector = embedding_model.embed_query(search_query)
    
    # 检索向量库
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=3,
        output_fields=["text"]
    )
    
    # 提取内容列表
    found_context = [hit['entity']['text'] for hit in results[0]]
    
    if not found_context:
        print("!!! [Retriever] 知识库未命中相关内容")
        
    return {"context": found_context}