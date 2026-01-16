import os
from typing import List, Dict, Any
from elasticsearch import Elasticsearch, helpers
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class DualWriter:
    def __init__(self):
        # 1. 初始化 OpenAI Embedding (用于 Milvus)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 2. 初始化 Elasticsearch 客户端
        self.es = Elasticsearch([os.getenv("ES_URL", "http://localhost:9200")])
        self.es_index = "deep_learner_knowledge"
        
        # 3. 初始化 Milvus 连接
        connections.connect("default", host="localhost", port="19530")
        self.milvus_collection_name = "deep_learner_vectors"
        self._setup_milvus()

    def _setup_milvus(self):
        """确保 Milvus 集合存在并配置好 Schema"""
        if not self.milvus_collection_name in [c for c in connections.list_connections()]:
            # 定义字段：ID, 向量, 原始文本(可选), 元数据
            fields = [
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536), # OpenAI 3-small 维度
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535)
            ]
            schema = CollectionSchema(fields, "Deep-Learner 知识库向量索引")
            self.collection = Collection(self.milvus_collection_name, schema)
            # 创建 HNSW 索引提速
            self.collection.create_index("vector", {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 8, "efConstruction": 64}})
        else:
            self.collection = Collection(self.milvus_collection_name)

    def write_all(self, data_list: List[Dict[str, Any]]):
        """
        核心分流逻辑：将 Spark 传来的字典列表同步写入两库
        """
        print(f"--- [DualWriter] 开始同步写入，共计 {len(data_list)} 条记录 ---")
        
        # 提取内容用于批量生成向量
        texts = [item['content'] for item in data_list]
        vectors = self.embeddings.embed_documents(texts)

        # A. 写入 Milvus
        milvus_data = [
            [item['doc_id'] for item in data_list],
            vectors,
            texts
        ]
        self.collection.upsert(milvus_data)
        self.collection.flush()
        print("✅ Milvus 写入成功")

        # B. 写入 Elasticsearch (使用 bulk 批量操作提速)
        es_actions = [
            {
                "_index": self.es_index,
                "_id": item['doc_id'],
                "_source": {
                    "content": item['content'],
                    "main_topic": item['h1'],
                    "sub_topic": item['h2'],
                    "source": item['source']
                }
            }
            for item in data_list
        ]
        helpers.bulk(self.es, es_actions)
        print("✅ Elasticsearch 写入成功")