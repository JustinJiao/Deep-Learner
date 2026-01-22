import os
from typing import List, Dict
from retrieval.base import SearchResult
from retrieval.vector_retriever import VectorRetriever
from retrieval.keyword_retriever import KeywordRetriever
from retrieval.reranker import Reranker

class RetrievalPipeline:
    def __init__(
        self, 
        vector_retriever: VectorRetriever, 
        keyword_retriever: KeywordRetriever, 
        reranker: Reranker = None
    ):
        """
        初始化检索流水线
        """
        self.vector_r = vector_retriever
        self.keyword_r = keyword_retriever
        self.reranker = reranker
        
        # 从环境变量加载 RRF 平滑常数，默认 60
        self.rrf_k = int(os.getenv("RRF_K", 60))

    def _rrf_fusion(self, v_results: List[SearchResult], k_results: List[SearchResult]) -> List[SearchResult]:
        r"""
        实现 RRF (Reciprocal Rank Fusion) 融合算法。
        该算法只依赖排名而不依赖原始分数，解决了 Milvus 与 ES 分数不可比的问题。
        
        公式：$$RRFscore(d) = \sum_{r \in R} \frac{1}{k + rank(d, r)}$$
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, SearchResult] = {}

        # 1. 处理向量检索排名 (已按分数由高到低排序)
        for rank, res in enumerate(v_results, start=1):
            # 使用 MD5 ID 作为唯一键对齐
            rrf_scores[res.id] = rrf_scores.get(res.id, 0) + 1.0 / (self.rrf_k + rank)
            doc_map[res.id] = res

        # 2. 处理关键词检索排名
        for rank, res in enumerate(k_results, start=1):
            rrf_scores[res.id] = rrf_scores.get(res.id, 0) + 1.0 / (self.rrf_k + rank)
            if res.id not in doc_map:
                doc_map[res.id] = res

        # 3. 按 RRF 融合总分从大到小排列 (倒序)
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_candidates = []
        for doc_id, combined_score in sorted_items:
            doc = doc_map[doc_id]
            doc.score = combined_score  # 更新为混合分数
            final_candidates.append(doc)
            
        return final_candidates

    def run(self, query: str) -> List[SearchResult]:
        """
        执行完整的 Two-stage 检索任务
        """
        # 从 .env 读取召回和最终保留的数量配置
        recall_top_k = int(os.getenv("RECALL_TOP_K", 50))
        final_top_k = int(os.getenv("FINAL_TOP_K", 5))

        # --- Stage 1: 粗排 (双路召回) ---
        # 并行或顺序调用 Milvus 和 ES
        v_res = self.vector_r.search(query, top_k=recall_top_k)
        k_res = self.keyword_r.search(query, top_k=recall_top_k)
        
        # 执行 RRF 融合
        candidates = self._rrf_fusion(v_res, k_res)
        
        # --- Stage 2: 精排 (重排序) ---
        # 如果配置了 Reranker，则利用深度学习模型对候选集进行二次打分
        if self.reranker and candidates:
            # 仅对融合后的候选集进行昂贵的重排计算
            return self.reranker.rerank(query, candidates, top_n=final_top_k)
        
        # 若未配置 Reranker，直接返回 RRF 评分最高的结果
        return candidates[:final_top_k]