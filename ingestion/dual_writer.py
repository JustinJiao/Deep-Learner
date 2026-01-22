import os
from config.settings import AppConfig
from services.embedding_service import EmbeddingService # 🌟 引用已解耦的服务

class DualWriter:
    def __init__(self, milvus_col, es_client):
        self.collection = milvus_col
        self.es = es_client
        self.es_index = AppConfig.ES_INDEX
        # 🌟 不再在这里配置 OpenAI 详情，直接使用统一服务
        self.emb_service = EmbeddingService()

    def write_all(self, data_list):
        if not data_list: return

        # 1. 调用统一的 Embedding 服务（批量处理）
        contents = [d['content'] for d in data_list]
        try:
            vectors = self.emb_service.get_batch_embeddings(contents)
            
            if vectors:
                # 🌟 关键修改：组装数据，确保 metadata 作为一个 JSON 列写入
                milvus_data = [
                    [d['doc_id'] for d in data_list],      # doc_id
                    vectors,                               # vector
                    [d['content'] for d in data_list],    # content
                    [d['metadata'] for d in data_list]     # metadata (JSON 格式)
                ]
                
                self.collection.upsert(milvus_data)
                self.collection.flush() 
                print(f"✅ Milvus 成功写入 {len(data_list)} 条向量数据")
            
        except Exception as e:
            print(f"❌ Milvus 写入失败: {e}")

        # 2. ES Bulk 写入
        from elasticsearch import helpers
        actions = [{
            "_index": self.es_index,
            "_id": d["doc_id"],
            "_source": {
                "content": d["content"],
                "metadata": d["metadata"] # ES 也保持相同的 metadata 结构
            }
        } for d in data_list]
        
        helpers.bulk(self.es, actions)
        print(f"✅ Elasticsearch 文本写入成功")