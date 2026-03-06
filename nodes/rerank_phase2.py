import time
from pathlib import Path

from config.settings import AppConfig
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


def _extract_source_name(doc: dict) -> str:
    metadata = doc.get("metadata", {}) or {}
    source_raw = (
        metadata.get("source")
        or doc.get("title")
        or doc.get("id")
        or "Unknown Document"
    )
    source_text = str(source_raw).strip()
    return Path(source_text).name or source_text or "Unknown Document"


def _coverage_window_top_k() -> int:
    configured = int(AppConfig.RETRIEVAL_SOURCE_COVERAGE_WINDOW_TOP_K)
    if configured > 0:
        return configured
    compose_top_k = int(AppConfig.RUNTIME_COMPOSE_CONTEXT_TOP_K)
    return compose_top_k if compose_top_k > 0 else 1


def _dedupe_docs_by_id(docs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for doc in docs:
        doc_id = str(doc.get("id", "")).strip()
        if not doc_id:
            continue
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc)
    return out


def _enforce_source_coverage(
    ranked_docs: list[dict],
    candidates: list[dict],
) -> tuple[list[dict], int]:
    if not bool(AppConfig.RETRIEVAL_ENFORCE_SOURCE_COVERAGE):
        return ranked_docs, 0

    min_coverage = int(AppConfig.RETRIEVAL_MIN_SOURCE_COVERAGE)
    if min_coverage <= 0:
        return ranked_docs, 0

    augmented = _dedupe_docs_by_id(list(ranked_docs))
    if not augmented:
        return augmented, 0

    window_k = max(1, min(_coverage_window_top_k(), len(augmented)))
    window_sources = {_extract_source_name(doc) for doc in augmented[:window_k]}
    if len(window_sources) >= min_coverage:
        return augmented, 0

    source_representative: dict[str, dict] = {}
    for doc in augmented:
        source = _extract_source_name(doc)
        source_representative.setdefault(source, doc)
    for cand in candidates:
        source = _extract_source_name(cand)
        source_representative.setdefault(source, cand)

    max_possible = min(min_coverage, len(source_representative))
    if len(window_sources) >= max_possible:
        return augmented, 0

    supplemental: list[dict] = []
    for source, rep in source_representative.items():
        if source in window_sources:
            continue
        supplemental.append(rep)
        if len(window_sources) + len(supplemental) >= max_possible:
            break

    added = 0
    insert_at = 1
    for cand in supplemental:
        cand_id = str(cand.get("id", "")).strip()
        if cand_id:
            augmented = [d for d in augmented if str(d.get("id", "")).strip() != cand_id]
        pos = min(insert_at, len(augmented))
        augmented.insert(pos, cand)
        insert_at += 1
        added += 1
        window_k = max(1, min(_coverage_window_top_k(), len(augmented)))
        window_sources = {_extract_source_name(doc) for doc in augmented[:window_k]}
        if len(window_sources) >= max_possible:
            break

    return _dedupe_docs_by_id(augmented), added


def rerank_phase2_node(state: AgentState) -> AgentState:
    query = state.get("retrieval_query") or state.get("query", "")
    rerank_top_n = max(1, int(AppConfig.PHASE2_RERANK_TOP_N))
    candidates = state.get("phase2_candidates", [])[:rerank_top_n]
    reranker = _get_reranker()

    if reranker is None:
        reranked_docs = candidates
        rerank_enabled = False
    else:
        input_results = _to_search_results(candidates)
        reranked = reranker.rerank(
            query, input_results, top_n=min(rerank_top_n, len(input_results))
        )
        reranked_docs = _to_context_docs(reranked)
        rerank_enabled = True

    reranked_docs, source_coverage_added = _enforce_source_coverage(
        ranked_docs=reranked_docs,
        candidates=candidates,
    )
    source_coverage_count = len({_extract_source_name(doc) for doc in reranked_docs})

    state["phase2_reranked"] = _to_trace_docs(reranked_docs)
    state["context_pool"] = reranked_docs
    state["context_source"] = "phase2"
    # 释放 phase2 检索中间结果，降低后续状态复制成本。
    state.pop("phase2_candidates", None)

    state.setdefault("steps_log", []).append(
        StepLog(
            node="rerank_phase2",
            info={
                "state": {
                    "query_preview": clip_text(query, 180),
                },
                "memory": {
                    "rerank_top_n": rerank_top_n,
                    "rerank_input_count": len(candidates),
                    "rerank_output_count": len(reranked_docs),
                    "rerank_enabled": rerank_enabled,
                    "source_coverage_count": source_coverage_count,
                    "source_coverage_added": source_coverage_added,
                    "context_pool_preview": preview_docs(reranked_docs),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
