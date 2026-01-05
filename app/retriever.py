from pymilvus import MilvusClient
from .state import AgentState
from services.llm_service import embedding_model

# 初始化 Milvus 客户端 (与 Ingest 保持一致)
COLLECTION_NAME = "deep_learner_docs"
client = MilvusClient("./database/milvus_demo.db")

def retriever_node(state: AgentState):
    search_query = state['plan']
    print(f"--- [Retriever] 正在从 Milvus 检索: {search_query} ---")
    
    # 1. 将搜索计划向量化
    query_vector = embedding_model.embed_query(search_query)
    
    # 2. 执行向量搜索
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=3,  # 取 Top 3
        output_fields=["text"]
    )
    
    # 3. 提取文本内容
    found_context = []
    for hit in results[0]:
        found_context.append(hit['entity']['text'])
        
    if not found_context:
        found_context = ["未找到相关本地知识，请引导学生参考官方文档。"]
        
    print(f"--- [Retriever] 检索完成，找到 {len(found_context)} 条相关背景 ---")
    return {"context": found_context}