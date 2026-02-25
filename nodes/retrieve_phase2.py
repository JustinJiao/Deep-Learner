import time
from typing import Dict

from config.settings import AppConfig
from core.state import AgentState, StepLog
from nodes.log_utils import clip_text, preview_docs
from tools.retrieve_tool.base import SearchResult
from tools.retrieve_tool.keyword import KeywordRetriever
from tools.retrieve_tool.vector import VectorRetriever

_VECTOR_RETRIEVER: VectorRetriever | None = None
_KEYWORD_RETRIEVER: KeywordRetriever | None = None


def _get_vector_retriever() -> VectorRetriever:
    global _VECTOR_RETRIEVER
    if _VECTOR_RETRIEVER is None:
        _VECTOR_RETRIEVER = VectorRetriever()
    return _VECTOR_RETRIEVER


def _get_keyword_retriever() -> KeywordRetriever:
    global _KEYWORD_RETRIEVER
    if _KEYWORD_RETRIEVER is None:
        _KEYWORD_RETRIEVER = KeywordRetriever()
    return _KEYWORD_RETRIEVER


def _rrf_fusion(
    v_results: list[SearchResult],
    k_results: list[SearchResult],
) -> list[SearchResult]:
    rrf_k = int(getattr(AppConfig, "RRF_K", 60))
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, SearchResult] = {}

    for rank, res in enumerate(v_results, start=1):
        rrf_scores[res.id] = rrf_scores.get(res.id, 0.0) + 1.0 / (rrf_k + rank)
        doc_map[res.id] = res

    for rank, res in enumerate(k_results, start=1):
        rrf_scores[res.id] = rrf_scores.get(res.id, 0.0) + 1.0 / (rrf_k + rank)
        if res.id not in doc_map:
            doc_map[res.id] = res

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    merged: list[SearchResult] = []
    for doc_id, fused_score in ranked:
        item = doc_map[doc_id]
        item.score = float(fused_score)
        merged.append(item)
    return merged


def retrieve_phase2_node(state: AgentState) -> AgentState:
    original_query = str(state.get("query", "")).strip()
    retrieval_query = str(
        state.get("retrieval_query") or state.get("resolved_query") or original_query
    ).strip()
    vector_top_k = int(getattr(AppConfig, "PHASE2_VECTOR_TOP_K", 80))
    keyword_top_k = int(getattr(AppConfig, "PHASE2_KEYWORD_TOP_K", 80))

    vector_hits = _get_vector_retriever().search(retrieval_query, top_k=vector_top_k)
    keyword_hits = _get_keyword_retriever().search(retrieval_query, top_k=keyword_top_k)
    merged = _rrf_fusion(vector_hits, keyword_hits)

    phase2_candidates: list[dict] = []
    for r in merged:
        title = None
        if isinstance(r.metadata, dict):
            title = r.metadata.get("title")
        phase2_candidates.append(
            {
                "id": r.id,
                "title": title or r.source_type or "Untitled",
                "content": r.content,
                "score": float(r.score),
                "metadata": r.metadata,
                "source_type": r.source_type,
            }
        )

    state["phase2_candidates"] = phase2_candidates

    state.setdefault("steps_log", []).append(
        StepLog(
            node="retrieve_phase2",
            info={
                "state": {
                    "query_preview": clip_text(original_query, 180),
                    "retrieval_query_preview": clip_text(retrieval_query, 180),
                    "used_rewritten_query": retrieval_query != original_query,
                },
                "memory": {
                    "vector_top_k": vector_top_k,
                    "keyword_top_k": keyword_top_k,
                    "vector_hits": len(vector_hits),
                    "keyword_hits": len(keyword_hits),
                    "merged_count": len(phase2_candidates),
                    "candidate_preview": preview_docs(phase2_candidates),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
