import time

from core.state import AgentState, StepLog
from nodes.log_utils import clip_text, preview_docs
from tools.retrieve_tool.base import SearchResult
from tools.retrieve_tool.rerank import Reranker

_RERANKER: Reranker | None = None


def _get_reranker() -> Reranker | None:
    global _RERANKER
    if _RERANKER is None:
        try:
            _RERANKER = Reranker()
        except Exception:
            _RERANKER = None
    return _RERANKER


def _to_search_results(candidates: list[dict]) -> list[SearchResult]:
    results: list[SearchResult] = []
    for item in candidates:
        results.append(
            SearchResult(
                id=item.get("id", ""),
                content=item.get("content", ""),
                score=float(item.get("score", 0.0)),
                metadata=item.get("metadata", {}) or {},
                source_type=item.get("source_type", ""),
            )
        )
    return results


def _to_context_docs(results: list[SearchResult]) -> list[dict]:
    docs: list[dict] = []
    for r in results:
        title = None
        if isinstance(r.metadata, dict):
            title = r.metadata.get("title")
        docs.append(
            {
                "id": r.id,
                "title": title or r.source_type or "Untitled",
                "content": r.content,
                "score": float(r.score),
                "metadata": r.metadata,
                "source_type": r.source_type,
            }
        )
    return docs


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_trace_docs(docs: list[dict]) -> list[dict]:
    traced: list[dict] = []
    for item in docs:
        traced.append(
            {
                "id": item.get("id", ""),
                "title": item.get("title", ""),
                "score": _safe_float(item.get("score"), 0.0),
            }
        )
    return traced


def rerank_phase1_node(state: AgentState) -> AgentState:
    query = state.get("query", "")
    candidates = state.get("phase1_candidates", [])[:20]
    reranker = _get_reranker()

    if reranker is None:
        reranked_docs = candidates
        rerank_enabled = False
    else:
        input_results = _to_search_results(candidates)
        reranked = reranker.rerank(query, input_results, top_n=min(20, len(input_results)))
        reranked_docs = _to_context_docs(reranked)
        rerank_enabled = True

    state["phase1_reranked"] = _to_trace_docs(reranked_docs)
    state["context_pool"] = reranked_docs
    state["context_source"] = "phase1"
    # 释放大字段，避免后续节点 deepcopy 成本被重复支付。
    state.pop("phase1_candidates", None)

    state.setdefault("steps_log", []).append(
        StepLog(
            node="rerank_phase1",
            info={
                "state": {
                    "query_preview": clip_text(query, 180),
                },
                "memory": {
                    "rerank_input_count": len(candidates),
                    "rerank_output_count": len(reranked_docs),
                    "rerank_enabled": rerank_enabled,
                    "context_pool_preview": preview_docs(reranked_docs),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
