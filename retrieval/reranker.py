import os
from typing import List
from sentence_transformers import CrossEncoder
from retrieval.base import SearchResult

class Reranker:
    def __init__(self):
        # 从 .env 读取模型路径和运行设备 (cpu/mps/cuda)
        model_path = os.getenv("RERANK_MODEL_PATH")
        device = os.getenv("RERANK_DEVICE", "cpu")
        if not model_path:
            raise ValueError("❌ .env 中缺失 RERANK_MODEL_PATH 配置")
        self.model = CrossEncoder(model_path, device=device)

    def rerank(self, query: str, candidates: List[SearchResult], top_n: int) -> List[SearchResult]:
        if not candidates: return []

        pairs = [[query, cand.content] for cand in candidates]
        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i].score = float(score)

        return sorted(candidates, key=lambda x: x.score, reverse=True)[:top_n]