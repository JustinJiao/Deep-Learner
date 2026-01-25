from typing import List
from sentence_transformers import CrossEncoder
from retrieval.base import SearchResult
from config.settings import AppConfig

class Reranker:
    def __init__(self):
        if not AppConfig.RERANK_MODEL_PATH:
            raise ValueError("❌ AppConfig 中缺失 RERANK_MODEL_PATH 配置")
        
        self.model = CrossEncoder(
            AppConfig.RERANK_MODEL_PATH, 
            device=AppConfig.RERANK_DEVICE
        )

    def rerank(self, query: str, candidates: List[SearchResult], top_n: int) -> List[SearchResult]:
        if not candidates: return []

        pairs = [[query, cand.content] for cand in candidates]
        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i].score = float(score)

        return sorted(candidates, key=lambda x: x.score, reverse=True)[:top_n]