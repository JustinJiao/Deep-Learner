import os
from config.settings import AppConfig, ResourceFactory
from elasticsearch import helpers

class DualWriter:
    def __init__(self, milvus_col=None, es_client=None):
        # 🌟 优化：如果初始化没传，直接去工厂拿单例
        self.collection = milvus_col or ResourceFactory.get_milvus_collection()
        self.es = es_client or ResourceFactory.get_es_client()
        self.es_index = AppConfig.ES_INDEX
        
        # 🌟 关键：从工厂获取解耦后的 Embedding 服务
        self.emb_service = ResourceFactory.get_embedding_service()

    def write_all(self, data_list):
        if not data_list: return

        # 1. 批量向量化 (使用更新后的接口名 embed_documents)
        contents = [d['content'] for d in data_list]
        try:
            vectors = self.emb_service.embed_documents(contents)
            
            if vectors:
                milvus_data = [
                    [d['doc_id'] for d in data_list],      
                    vectors,                               
                    [d['content'] for d in data_list],    
                    [d['metadata'] for d in data_list]     
                ]
                
                self.collection.upsert(milvus_data)
                self.collection.flush() 
                print(f"✅ Milvus 成功写入 {len(data_list)} 条向量数据")
            
        except Exception as e:
            print(f"❌ Milvus 写入失败: {e}")

        # 2. ES Bulk 写入
        actions = [{
            "_index": self.es_index,
            "_id": d["doc_id"],
            "_source": {
                "content": d["content"],
                "metadata": d["metadata"] 
            }
        } for d in data_list]
        
        helpers.bulk(self.es, actions)
        print(f"✅ Elasticsearch 文本写入成功")