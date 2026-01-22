import os
from dotenv import load_dotenv
from pymilvus import connections, Collection
from elasticsearch import Elasticsearch

# 按照你要求的绝对导入路径
from services.embedding_service import EmbeddingService
from retrieval.vector_retriever import VectorRetriever
from retrieval.keyword_retriever import KeywordRetriever
from retrieval.reranker import Reranker
from retrieval.pipeline import RetrievalPipeline

def setup_clients():
    """初始化数据库客户端"""
    load_dotenv()
    
    # 1. 连接 Milvus
    connections.connect(
        alias="default", 
        host=os.getenv("MILVUS_HOST", "localhost"), 
        port=os.getenv("MILVUS_PORT", "19530")
    )
    milvus_coll = Collection(os.getenv("MILVUS_COLLECTION_NAME"))
    milvus_coll.load() # 必须加载到内存才能搜索
    
    # 2. 连接 Elasticsearch
    es_client = Elasticsearch([f"http://{os.getenv('ES_HOST')}:{os.getenv('ES_PORT')}"])
    
    return milvus_coll, es_client

def run_integration_test(query: str):
    print(f"\n🚀 开始集成测试 | 提问: {query}")
    print("-" * 50)

    # 1. 初始化模型服务
    embed_service = EmbeddingService() # 默认使用 .env 中的 provider
    
    # 2. 初始化各级检索器
    milvus_coll, es_client = setup_clients()
    v_retriever = VectorRetriever(milvus_coll, embed_service)
    k_retriever = KeywordRetriever(es_client)
    
    # 3. 初始化重排序器 (由于加载模型较慢，建议在 Pipeline 中按需开启)
    try:
        my_reranker = Reranker()
        print("✅ Reranker 模型加载成功")
    except Exception as e:
        print(f"⚠️ Reranker 加载失败 (可能是缺少模型文件): {e}")
        my_reranker = None

    # 4. 组装 Pipeline
    pipeline = RetrievalPipeline(
        vector_retriever=v_retriever,
        keyword_retriever=k_retriever,
        reranker=my_reranker
    )

    # 5. 执行测试
    results = pipeline.run(query)

    # 6. 结果展示
    print(f"\n🎯 最终召回结果 (Top {len(results)}):")
    for i, res in enumerate(results, 1):
        print(f"[{i}] 分数: {res.score:.4f} | 来源: {res.source_type}")
        print(f"    ID: {res.id}")
        # 打印部分元数据，验证 ingestion 是否成功对齐
        file_name = res.metadata.get('filename', '未知文件')
        print(f"    源文件: {file_name}")
        print(f"    内容摘要: {res.content[:150].strip()}...")
        print("-" * 30)

if __name__ == "__main__":
    # 建议使用你在 Ingestion 阶段入库过的内容进行测试
    # 比如关于 Spark、代码随想录或算法的内容
    test_query = "transformer的核心？"
    
    try:
        run_integration_test(test_query)
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")