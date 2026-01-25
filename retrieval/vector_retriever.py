from typing import List
from retrieval.base import BaseRetriever, SearchResult
from config.settings import ResourceFactory

class VectorRetriever(BaseRetriever):
    def __init__(self, collection=None, embedding_service=None):
        super().__init__(name="MilvusVectorRetriever")
        # 🌟 统一资源获取
        self.collection = collection or ResourceFactory.get_milvus_collection()
        self.embedding_service = embedding_service or ResourceFactory.get_embedding_service()

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        # 调用解耦后的 embed_query 方法
        query_vector = self.embedding_service.embed_query(query)
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["content", "metadata"]
        )

        standard_results = []
        for hit in results[0]:
            standard_results.append(SearchResult(
                id=hit.id,
                content=hit.entity.get("content"),
                score=hit.score,
                metadata=hit.entity.get("metadata"),
                source_type="vector"
            ))
        return standard_results