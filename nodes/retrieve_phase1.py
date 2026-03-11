import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict
from pathlib import Path

from config.settings import AppConfig
from core.state import AgentState, StepLog
from nodes.log_utils import clip_text, preview_docs
from tools.retrieve_tool.base import SearchResult
from tools.retrieve_tool.keyword import KeywordRetriever
from tools.retrieve_tool.vector import VectorRetriever

_VECTOR_RETRIEVER: VectorRetriever | None = None
_KEYWORD_RETRIEVER: KeywordRetriever | None = None


def _get_retriever() -> VectorRetriever:
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
    rrf_k = int(AppConfig.RRF_K)
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


def _source_name_from_result(res: SearchResult) -> str:
    metadata = res.metadata if isinstance(res.metadata, dict) else {}
    source_raw = (
        metadata.get("source")
        or metadata.get("title")
        or res.source_type
        or res.id
        or "Unknown Document"
    )
    source_text = str(source_raw).strip()
    return (Path(source_text).name or source_text or "Unknown Document").lower()


def _filter_allowed_sources(results: list[SearchResult]) -> tuple[list[SearchResult], int]:
    if not bool(AppConfig.RETRIEVAL_RESTRICT_TO_ALLOWED_SOURCES):
        return results, 0
    allowed = set(AppConfig.RETRIEVAL_ALLOWED_SOURCES)
    if not allowed:
        return results, 0

    filtered = [r for r in results if _source_name_from_result(r) in allowed]
    removed = max(0, len(results) - len(filtered))
    return filtered, removed


def _normalize_retrieval_queries(state: AgentState, fallback_query: str) -> list[str]:
    raw = state.get("retrieval_queries", []) or []
    queries: list[str] = []
    seen: set[str] = set()

    for item in raw:
        text = " ".join(str(item or "").split()).strip()
        lowered = text.lower()
        if not text or lowered in seen:
            continue
        seen.add(lowered)
        queries.append(text)

    if queries:
        return queries

    fallback = " ".join(str(fallback_query or "").split()).strip()
    return [fallback] if fallback else [fallback_query]


def _result_to_candidate(res: SearchResult) -> dict:
    title = None
    if isinstance(res.metadata, dict):
        title = res.metadata.get("title")
    return {
        "id": res.id,
        "title": title or res.source_type or "Untitled",
        "content": res.content,
        "score": float(res.score),
        "metadata": dict(res.metadata or {}),
        "source_type": res.source_type,
    }


def _run_single_query_retrieval(
    retrieval_query: str,
    vector_top_k: int,
    keyword_top_k: int,
    keep_top_m: int,
) -> dict:
    raw_vector_hits = _get_retriever().search(retrieval_query, top_k=vector_top_k)
    raw_keyword_hits = _get_keyword_retriever().search(retrieval_query, top_k=keyword_top_k)
    vector_hits, vector_filtered_out = _filter_allowed_sources(raw_vector_hits)
    keyword_hits, keyword_filtered_out = _filter_allowed_sources(raw_keyword_hits)
    merged = _rrf_fusion(vector_hits, keyword_hits)
    if keep_top_m > 0:
        merged = merged[:keep_top_m]
    return {
        "query": retrieval_query,
        "raw_vector_hits": raw_vector_hits,
        "raw_keyword_hits": raw_keyword_hits,
        "vector_hits": vector_hits,
        "keyword_hits": keyword_hits,
        "vector_filtered_out": vector_filtered_out,
        "keyword_filtered_out": keyword_filtered_out,
        "merged": merged,
    }


def _aggregate_multi_query_results(query_runs: list[dict]) -> list[dict]:
    rrf_k = int(AppConfig.RRF_K)
    coverage_weight = float(AppConfig.RETRIEVAL_MULTI_QUERY_COVERAGE_WEIGHT)
    aggregated: Dict[str, dict] = {}

    for run in query_runs:
        query_text = str(run.get("query", ""))
        merged: list[SearchResult] = run.get("merged", []) or []
        for rank, res in enumerate(merged, start=1):
            if not str(res.id or "").strip():
                continue
            rec = aggregated.get(res.id)
            if rec is None:
                rec = _result_to_candidate(res)
                rec["metadata"]["__mq_coverage_count"] = 0
                rec["metadata"]["__mq_best_rank"] = rank
                rec["metadata"]["__mq_matched_queries"] = []
                rec["_mq_query_set"] = set()
                rec["score"] = 0.0
                aggregated[res.id] = rec

            rec["score"] += 1.0 / (rrf_k + rank)
            rec["metadata"]["__mq_best_rank"] = min(
                int(rec["metadata"].get("__mq_best_rank", rank)),
                rank,
            )
            matched: set[str] = rec["_mq_query_set"]
            if query_text and query_text not in matched:
                matched.add(query_text)
                rec["metadata"]["__mq_matched_queries"].append(query_text)

    out: list[dict] = []
    for rec in aggregated.values():
        matched = rec.pop("_mq_query_set", set())
        coverage_count = len(matched)
        rec["metadata"]["__mq_coverage_count"] = coverage_count
        prior_score = float(rec.get("score", 0.0)) + (coverage_weight * max(0, coverage_count - 1))
        rec["metadata"]["__mq_prior_score"] = prior_score
        rec["score"] = prior_score
        out.append(rec)

    return sorted(out, key=lambda x: float(x.get("score", 0.0)), reverse=True)


def _build_query_route_records(
    query_runs: list[dict],
    per_route_top_k: int,
) -> list[dict]:
    routes: list[dict] = []
    keep_k = max(1, int(per_route_top_k))
    for route_idx, run in enumerate(query_runs):
        route_query = str(run.get("query", "")).strip()
        merged: list[SearchResult] = run.get("merged", []) or []
        route_candidates: list[dict] = []
        for rank, res in enumerate(merged[:keep_k], start=1):
            candidate = _result_to_candidate(res)
            metadata = dict(candidate.get("metadata", {}) or {})
            metadata["__mq_route_index"] = route_idx
            metadata["__mq_route_query"] = route_query
            metadata["__mq_route_rank"] = rank
            candidate["metadata"] = metadata
            route_candidates.append(candidate)
        routes.append(
            {
                "route_index": route_idx,
                "query": route_query,
                "candidate_count": len(merged),
                "candidates": route_candidates,
            }
        )
    return routes


def retrieve_phase1_node(state: AgentState) -> AgentState:
    original_query = str(state.get("query", "")).strip()
    retrieval_query = str(
        state.get("retrieval_query") or state.get("resolved_query") or original_query
    ).strip()
    retrieval_queries = _normalize_retrieval_queries(state=state, fallback_query=retrieval_query)
    # Auto-enable async multi-query retrieval whenever multiple sub-queries are present.
    multi_query_enabled = len(retrieval_queries) > 1

    query_runs: list[dict] = []
    if multi_query_enabled:
        vector_top_k = int(AppConfig.PHASE1_PER_QUERY_VECTOR_TOP_K)
        keyword_top_k = int(AppConfig.PHASE1_PER_QUERY_KEYWORD_TOP_K)
        keep_top_m = int(AppConfig.PHASE1_PER_QUERY_KEEP_TOP_M)
        workers = max(
            1,
            min(int(AppConfig.RETRIEVAL_MULTI_QUERY_PARALLEL_WORKERS), len(retrieval_queries)),
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    _run_single_query_retrieval,
                    retrieval_query=q,
                    vector_top_k=vector_top_k,
                    keyword_top_k=keyword_top_k,
                    keep_top_m=keep_top_m,
                ): idx
                for idx, q in enumerate(retrieval_queries)
            }
            ordered: Dict[int, dict] = {}
            for future in as_completed(future_map):
                idx = future_map[future]
                ordered[idx] = future.result()
        query_runs = [ordered[i] for i in sorted(ordered.keys())]
        phase1_candidates = _aggregate_multi_query_results(query_runs)
    else:
        vector_top_k = int(AppConfig.PHASE1_VECTOR_TOP_K)
        keyword_top_k = int(AppConfig.PHASE1_KEYWORD_TOP_K)
        single_run = _run_single_query_retrieval(
            retrieval_query=retrieval_query,
            vector_top_k=vector_top_k,
            keyword_top_k=keyword_top_k,
            keep_top_m=0,
        )
        query_runs = [single_run]
        phase1_candidates = [_result_to_candidate(r) for r in single_run["merged"]]

    phase1_query_routes: list[dict] = []
    if multi_query_enabled:
        phase1_query_routes = _build_query_route_records(
            query_runs=query_runs,
            per_route_top_k=int(AppConfig.RETRIEVAL_MULTI_QUERY_ROUTE_TOP_K),
        )

    state["phase1_candidates"] = phase1_candidates
    state["phase1_query_routes"] = phase1_query_routes

    query_stats: list[dict] = []
    vector_hits_raw = 0
    keyword_hits_raw = 0
    vector_hits_kept = 0
    keyword_hits_kept = 0
    vector_filtered_out = 0
    keyword_filtered_out = 0
    for run in query_runs:
        vector_hits_raw += len(run.get("raw_vector_hits", []) or [])
        keyword_hits_raw += len(run.get("raw_keyword_hits", []) or [])
        vector_hits_kept += len(run.get("vector_hits", []) or [])
        keyword_hits_kept += len(run.get("keyword_hits", []) or [])
        vector_filtered_out += int(run.get("vector_filtered_out", 0) or 0)
        keyword_filtered_out += int(run.get("keyword_filtered_out", 0) or 0)
        query_stats.append(
            {
                "query_preview": clip_text(str(run.get("query", "")), 150),
                "vector_hits_raw": len(run.get("raw_vector_hits", []) or []),
                "keyword_hits_raw": len(run.get("raw_keyword_hits", []) or []),
                "vector_hits": len(run.get("vector_hits", []) or []),
                "keyword_hits": len(run.get("keyword_hits", []) or []),
                "merged_count": len(run.get("merged", []) or []),
            }
        )

    state.setdefault("steps_log", []).append(
        StepLog(
            node="retrieve_phase1",
            info={
                "state": {
                    "query_preview": clip_text(original_query, 180),
                    "retrieval_query_preview": clip_text(retrieval_query, 180),
                    "retrieval_queries": [clip_text(q, 160) for q in retrieval_queries],
                    "retrieval_query_count": len(retrieval_queries),
                    "used_rewritten_query": retrieval_query != original_query,
                },
                "memory": {
                    "multi_query_enabled": multi_query_enabled,
                    "vector_top_k": vector_top_k,
                    "keyword_top_k": keyword_top_k,
                    "vector_hits_raw": vector_hits_raw,
                    "keyword_hits_raw": keyword_hits_raw,
                    "vector_hits": vector_hits_kept,
                    "keyword_hits": keyword_hits_kept,
                    "vector_filtered_out": vector_filtered_out,
                    "keyword_filtered_out": keyword_filtered_out,
                    "query_stats": query_stats,
                    "query_route_count": len(phase1_query_routes),
                    "merged_count": len(phase1_candidates),
                    "candidate_count": len(phase1_candidates),
                    "candidate_preview": preview_docs(phase1_candidates),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
