import os
from typing import List
from elasticsearch import Elasticsearch
from retrieval.base import BaseRetriever, SearchResult

class KeywordRetriever(BaseRetriever):
    def __init__(self, es_client: Elasticsearch):
        super().__init__(name="ESKeywordRetriever")
        self.es = es_client
        self.index_name = os.getenv("ES_INDEX_NAME")

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        query_body = {
            "query": {"match": {"content": query}},
            "size": top_k
        }
        response = self.es.search(index=self.index_name, body=query_body)
        
        standard_results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            standard_results.append(SearchResult(
                id=hit['_id'],
                content=source.get("content"),
                score=hit['_score'],
                metadata=source.get("metadata", {}),
                source_type="keyword"
            ))
        return standard_results