# memory/ltm.py
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from config.factory import ResourceFactory
from config.settings import AppConfig


# memory/ltm.py

import time
from typing import List, Dict
from config.factory import ResourceFactory


class LTM:

    def __init__(self):
        self.collection = ResourceFactory.get_milvus_ltm_collection()
        self.embedding_service = ResourceFactory.get_embedding_service()

    def upsert(self, entries: List[Dict]):

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

            rows.append({
                "key": key,
                "vector": vector,
                "content": content,
                "type": entry_type,
                "score": score,
                "timestamp": timestamp,
            })

        self.collection.upsert(rows)
        self.collection.flush()

        return len(rows)

    def recall(self, query: str, top_k: int = 3):

        vector = self.embedding_service.embed_query(query)

        results = self.collection.search(
            data=[vector],
            anns_field="vector",
            param={
                "metric_type": "COSINE",
                "params": {"ef": 64}
            },
            limit=top_k,
            output_fields=["content"]
        )

        memories = []
        for hits in results:
            for hit in hits:
                memories.append(hit.entity.get("content"))

        return memories
