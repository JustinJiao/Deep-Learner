from __future__ import annotations

import time
from typing import Dict, List, Optional

from config.factory import ResourceFactory
from config.settings import AppConfig


class LTM:

    def __init__(self):
        self.collection = ResourceFactory.get_milvus_ltm_collection()
        self.embedding_service = ResourceFactory.get_embedding_service()

    def upsert(self, entries: List[Dict]) -> int:
        if not entries:
            return 0

        rows = []
        for entry in entries:
            key = entry["key"]
            content = entry["content"]
            entry_type = entry.get("type", "")
            score = float(entry.get("score", 0.0))
            timestamp = int(time.time())

            vector = self.embedding_service.embed_query(content)
            rows.append(
                {
                    "key": key,
                    "vector": vector,
                    "content": content,
                    "type": entry_type,
                    "score": score,
                    "timestamp": timestamp,
                }
            )

        self.collection.upsert(rows)
        self.collection.flush()
        return len(rows)

    def recall(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[str]:
        limit = top_k if top_k is not None else AppConfig.LTM_RECALL_TOP_K
        threshold = (
            min_score if min_score is not None else AppConfig.LTM_RECALL_THRESHOLD
        )
        threshold = max(-1.0, min(1.0, float(threshold)))
        if limit <= 0:
            return []

        prefix = (AppConfig.LTM_SEARCH_PREFIX or "").strip()
        query_for_embed = f"{prefix} {query}".strip() if prefix else query
        vector = self.embedding_service.embed_query(query_for_embed)

        results = self.collection.search(
            data=[vector],
            anns_field="vector",
            param={
                "metric_type": AppConfig.MILVUS_LTM_SEARCH_METRIC_TYPE,
                "params": {"ef": AppConfig.MILVUS_LTM_SEARCH_EF},
            },
            limit=limit,
            output_fields=["content"],
        )

        memories: List[str] = []
        seen = set()
        for hits in results:
            for hit in hits:
                score = float(getattr(hit, "distance", 0.0) or 0.0)
                if score < threshold:
                    continue

                content = hit.entity.get("content")
                if not content or content in seen:
                    continue

                seen.add(content)
                memories.append(content)

        return memories
