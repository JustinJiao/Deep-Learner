from config.settings import AppConfig
from config.factory import ResourceFactory
from elasticsearch import helpers
import os


class DualWriter:
    def __init__(self, milvus_col=None, es_client=None):
        # 🌟 优化：如果初始化没传，直接去工厂拿单例
        self.collection = milvus_col or ResourceFactory.get_milvus_collection()
        self.es = es_client or ResourceFactory.get_es_client()
        self.es_index = AppConfig.ES_INDEX

        # 🌟 关键：从工厂获取解耦后的 Embedding 服务
        self.emb_service = ResourceFactory.get_embedding_service()
        self.embed_batch_size = max(1, int(os.getenv("EMBED_BATCH_SIZE", "16")))

    def _iter_batches(self, data_list):
        for i in range(0, len(data_list), self.embed_batch_size):
            yield data_list[i : i + self.embed_batch_size]

    def write_all(self, data_list):
        if not data_list:
            return

        # 1. 批量向量化 + 分批写入 Milvus，避免单次 embedding 请求超 token 限制
        milvus_written = 0
        try:
            for batch in self._iter_batches(data_list):
                contents = [d["content"] for d in batch]
                vectors = self.emb_service.embed_documents(contents)
                if not vectors:
                    continue

                milvus_data = [
                    [d["doc_id"] for d in batch],
                    vectors,
                    [d["content"] for d in batch],
                    [d["metadata"] for d in batch],
                ]
                self.collection.upsert(milvus_data)
                milvus_written += len(batch)

            if milvus_written > 0:
                self.collection.flush()
                print(f"✅ Milvus 成功写入 {milvus_written} 条向量数据")
        except Exception as e:
            print(f"❌ Milvus 写入失败: {e}")

        # 2. ES Bulk 写入
        actions = [
            {
                "_index": self.es_index,
                "_id": d["doc_id"],
                "_source": {
                    "content": d["content"],
                    "metadata": d["metadata"],
                },
            }
            for d in data_list
        ]

        helpers.bulk(self.es, actions)
        print(f"✅ Elasticsearch 文本写入成功")
