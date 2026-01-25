from typing import List, Dict
from retrieval.base import SearchResult
from retrieval.vector_retriever import VectorRetriever
from retrieval.keyword_retriever import KeywordRetriever
from retrieval.reranker import Reranker
from config.settings import AppConfig

class RetrievalPipeline:
    def __init__(self, vector_r=None, keyword_r=None, reranker=None):
        """
        全自动化初始化，支持手动注入以供测试
        """
        self.vector_r = vector_r or VectorRetriever()
        self.keyword_r = keyword_r or KeywordRetriever()
        self.reranker = reranker or Reranker()
        self.rrf_k = AppConfig.RRF_K

    def _rrf_fusion(self, v_results: List[SearchResult], k_results: List[SearchResult]) -> List[SearchResult]:
        r"""
        实现 RRF 融合算法：
        $$RRFscore(d) = \sum_{r \in R} \frac{1}{k + rank(d, r)}$$
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, SearchResult] = {}

        for rank, res in enumerate(v_results, start=1):
            rrf_scores[res.id] = rrf_scores.get(res.id, 0) + 1.0 / (self.rrf_k + rank)
            doc_map[res.id] = res

        for rank, res in enumerate(k_results, start=1):
            rrf_scores[res.id] = rrf_scores.get(res.id, 0) + 1.0 / (self.rrf_k + rank)
            if res.id not in doc_map:
                doc_map[res.id] = res

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_candidates = []
        for doc_id, combined_score in sorted_items:
            doc = doc_map[doc_id]
            doc.score = combined_score
            final_candidates.append(doc)
            
        return final_candidates

    def run(self, query: str) -> List[SearchResult]:
        # 统一使用全局配置
        recall_top_k = AppConfig.RECALL_TOP_K
        final_top_k = AppConfig.FINAL_TOP_K

        # 1. 双路召回
        v_res = self.vector_r.search(query, top_k=recall_top_k)
        k_res = self.keyword_r.search(query, top_k=recall_top_k)
        
        # 2. RRF 融合
        candidates = self._rrf_fusion(v_res, k_res)
        
        # 3. 精排
        if self.reranker and candidates:
            return self.reranker.rerank(query, candidates, top_n=final_top_k)
        
        return candidates[:final_top_k]