import os
from openai import OpenAI
from .config import AppConfig

class DualWriter:
    def __init__(self, milvus_col, es_client):
        self.collection = milvus_col
        self.es = es_client
        self.es_index = AppConfig.ES_INDEX
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def write_all(self, data_list):
        if not data_list: return

        # 1. 为每个分块生成向量并组装 Milvus 数据
        contents = [d['content'] for d in data_list]
        try:
            # 批量调用 Embedding 模型 (text-embedding-3-small)
            response = self.client.embeddings.create(
                input=contents,
                model="text-embedding-3-small"
            )
            vectors = [item.embedding for item in response.data]
            
            # 组装符合 Milvus Schema 的列表数据
            milvus_data = [
                [d['doc_id'] for d in data_list],      # doc_id
                vectors,                               # vector
                [d['content'] for d in data_list],    # content
                [d['h1'] for d in data_list],         # h1
                [d['h2'] for d in data_list],         # h2
                [d['source'] for d in data_list]      # source
            ]
            
            # 正式写入 Milvus
            self.collection.upsert(milvus_data)
            # 强制刷新以使数据立即可见
            self.collection.flush() 
            print("✅ Milvus 向量写入并刷新成功")
            
        except Exception as e:
            print(f"❌ Milvus 写入失败: {e}")

        # 2. ES Bulk 写入保持不变
        from elasticsearch import helpers
        actions = [{
            "_index": self.es_index,
            "_id": d["doc_id"],
            "_source": d
        } for d in data_list]
        helpers.bulk(self.es, actions)
        print(f"✅ Elasticsearch 文本写入成功")