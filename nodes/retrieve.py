# nodes/retrieve.py

import time
from core.state import AgentState, StepLog
from tools.retrieve_tool.pipeline import RetrievalPipeline
from nodes.log_utils import clip_text, preview_docs

_PIPELINE: RetrievalPipeline | None = None


def _get_pipeline() -> RetrievalPipeline:
    global _PIPELINE
    # 测试中 monkeypatch RetrievalPipeline 时，避免复用旧实例导致跨用例污染
    if _PIPELINE is None or not isinstance(_PIPELINE, RetrievalPipeline):
        _PIPELINE = RetrievalPipeline()
    return _PIPELINE


def retrieve_node(state: AgentState) -> AgentState:
    pipeline = _get_pipeline()
    query = state.get("rewritten_query") or state["query"]

    results = pipeline.run(query)

    context_pool = []

    for r in results:
        title = None
        if isinstance(r.metadata, dict):
            title = r.metadata.get("title")

        context_pool.append({
            "id": r.id,
            "title": title or r.source_type or "Untitled",
            "content": r.content,
            "score": r.score,
            "metadata": r.metadata,
        })

    state["context_pool"] = context_pool

    state.setdefault("steps_log", []).append(
        StepLog(
            node="retrieve",
            info={
                "state": {
                    "query_preview": clip_text(state.get("query", ""), 180),
                    "retrieval_query_preview": clip_text(query, 180),
                    "used_rewritten_query": bool(state.get("rewritten_query")),
                },
                "memory": {
                    "context_pool_count": len(context_pool),
                    "context_pool_preview": preview_docs(context_pool),
                },
            },
            timestamp=time.time(),
        )
    )

    return state
