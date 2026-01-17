from pymilvus import connections, Collection
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv

load_dotenv()

def verify_ingestion():
    # 1. 检查 Elasticsearch
    es = Elasticsearch([os.getenv("ES_URL", "http://localhost:9200")])
    es_res = es.count(index="knowledge_base_index")
    print(f"--- [Elasticsearch 检查] ---")
    print(f"总记录数: {es_res['count']}")

    # 2. 检查 Milvus
    connections.connect("default", host="localhost", port="19530")
    collection = Collection("knowledge_vector_collection")
    collection.load() # 必须先加载到内存
    print(f"\n--- [Milvus 检查] ---")
    print(f"当前 Collection 名称: {collection.name}")
    print(f"总向量数 (num_entities): {collection.num_entities}")
    
    # 3. 随机抽样对比 ID 是否对齐
    # 在 ES 中找一条数据
    sample = es.search(index="knowledge_base_index", size=1)
    if sample['hits']['hits']:
        target_id = sample['hits']['hits'][0]['_id']
        print(f"\n--- [ID 对齐检查] ---")
        print(f"ES 采样 ID: {target_id}")
        
        # 在 Milvus 中按 ID 查询
        milvus_res = collection.query(expr=f"doc_id == '{target_id}'", output_fields=["doc_id"])
        if milvus_res and milvus_res[0]['doc_id'] == target_id:
            print(f"✅ ID 对齐验证成功！双路索引已锁死。")
        else:
            print(f"❌ ID 匹配失败，请检查写入逻辑。")

if __name__ == "__main__":
    verify_ingestion()