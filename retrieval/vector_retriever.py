from typing import List
from pymilvus import Collection
from retrieval.base import BaseRetriever, SearchResult
from services.embedding_service import EmbeddingService

class VectorRetriever(BaseRetriever):
    def __init__(self, collection: Collection, embedding_service: EmbeddingService):
        super().__init__(name="MilvusVectorRetriever")
        self.collection = collection
        self.embedding_service = embedding_service

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        query_vector = self.embedding_service.get_embedding(query)
        
        # 这里的 metric_type 也可以考虑放入 .env
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