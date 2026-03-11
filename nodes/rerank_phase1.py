import time
import re
from pathlib import Path

from config.settings import AppConfig
from core.state import AgentState, StepLog
from nodes.log_utils import clip_text, preview_docs
from nodes.rerank_route_features import route_consistency_delta
from tools.retrieve_tool.base import SearchResult
from tools.retrieve_tool.rerank import Reranker

_RERANKER: Reranker | None = None

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_GROUPED_NUM_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")
_REVENUE_QUERY_TERMS = {"revenue", "sales", "net", "segment"}
_BALANCE_QUERY_TERMS = {"balance", "year-end", "as of", "end of"}
_CASH_EQ_QUERY_TERMS = {"cash and cash equivalents", "cash equivalents"}
_REVENUE_HEADING_STRONG_PENALTY_TERMS = {
    "cost of sales",
    "foreign currency",
    "liabilities",
    "interest",
    "tax",
    "effective rate",
    "hedging",
    "investment",
    "allowance",
    "cash flow",
    "equity",
    "debt",
    "eps",
}
_BALANCE_HEADING_BOOST_TERMS = {
    "balance sheet",
    "balance sheets",
    "assets",
}
_BALANCE_HEADING_PENALTY_TERMS = {
    "stockholders",
    "comprehensive income",
    "operating activities",
    "cash provided by operating activities",
}
_REVENUE_HEADING_MEDIUM_PENALTY_TERMS = {
    "income",
    "expense",
}
_STOP_TERMS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "which",
    "what",
    "most",
    "latest",
    "fiscal",
    "year",
}


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


def _query_terms(query: str) -> set[str]:
    return {
        tok
        for tok in _TOKEN_RE.findall(str(query or "").lower())
        if len(tok) >= 3 and tok not in _STOP_TERMS
    }


def _query_calibration_delta(query: str, doc: dict) -> float:
    query_lower = str(query or "").lower()
    if not query_lower.strip():
        return 0.0

    content = str(doc.get("content", "") or "")
    content_lower = content.lower()
    heading = str((doc.get("metadata", {}) or {}).get("h2", "") or "").lower()

    q_terms = _query_terms(query_lower)
    d_terms = set(_TOKEN_RE.findall(content_lower))
    overlap = float(len(q_terms & d_terms)) / float(max(1, len(q_terms)))

    query_years = set(_YEAR_RE.findall(query_lower))
    year_hit = 1.0 if (query_years and any(y in content for y in query_years)) else 0.0
    numeric_density = min(1.0, float(len(_GROUPED_NUM_RE.findall(content))) / 8.0)

    anchor_hits = 0.0
    if "aws" in query_lower and "aws" in content_lower:
        anchor_hits += 1.0
    if "google cloud" in query_lower and "google cloud" in content_lower:
        anchor_hits += 1.0
    if ("azure" in query_lower or "microsoft cloud" in query_lower) and (
        "azure" in content_lower or "microsoft cloud" in content_lower
    ):
        anchor_hits += 1.0
    if any(term in query_lower for term in _REVENUE_QUERY_TERMS) and (
        "revenue" in content_lower or "net sales" in content_lower or "sales" in content_lower
    ):
        anchor_hits += 1.0
    if "cash and cash equivalents" in query_lower and "cash and cash equivalents" in content_lower:
        anchor_hits += 1.0
    anchor_score = min(1.0, anchor_hits / 3.0)

    penalty = 0.0
    asks_revenue = any(term in query_lower for term in _REVENUE_QUERY_TERMS)
    heading_is_revenue = ("revenue" in heading) or ("net sales" in heading) or ("cloud" in heading)
    if "cost of sales" in heading:
        heading_is_revenue = False
    if asks_revenue and not heading_is_revenue:
        if any(term in heading for term in _REVENUE_HEADING_STRONG_PENALTY_TERMS):
            penalty = 0.34
        elif any(term in heading for term in _REVENUE_HEADING_MEDIUM_PENALTY_TERMS):
            penalty = 0.22

    balance_boost = 0.0
    asks_balance = any(term in query_lower for term in _BALANCE_QUERY_TERMS)
    asks_cash_eq = any(term in query_lower for term in _CASH_EQ_QUERY_TERMS)
    if asks_balance:
        if any(term in heading for term in _BALANCE_HEADING_BOOST_TERMS):
            balance_boost += 0.14
        if "consolidated balance sheets" in content_lower:
            balance_boost += 0.12
    if asks_cash_eq:
        if "cash and cash equivalents" in content_lower:
            balance_boost += 0.14
        if "cash, cash equivalents, and short term marketable securities" in content_lower:
            # broader metric than pure cash and cash equivalents; keep but slightly down-weight.
            penalty += 0.08
        if "cash, cash equivalents, and marketable securities" in content_lower:
            penalty += 0.06
    if asks_balance and any(term in heading for term in _BALANCE_HEADING_PENALTY_TERMS):
        penalty += 0.12

    delta = (
        (overlap * 0.22)
        + (year_hit * 0.08)
        + (numeric_density * 0.12)
        + (anchor_score * 0.16)
        + balance_boost
        - penalty
    )
    delta += route_consistency_delta(query=query_lower, doc=doc)
    return float(delta)


def _apply_query_calibration(docs: list[dict], query: str) -> tuple[list[dict], int]:
    if not docs:
        return [], 0
    adjusted: list[dict] = []
    touched = 0
    for doc in docs:
        delta = _query_calibration_delta(query=query, doc=doc)
        if delta != 0.0:
            doc["score"] = float(doc.get("score", 0.0)) + float(delta)
            touched += 1
        adjusted.append(doc)
    adjusted.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return adjusted, touched

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


def _blend_multi_query_coverage_score(
    docs: list[dict],
    multi_query_enabled: bool,
) -> tuple[list[dict], int]:
    if not multi_query_enabled:
        return docs, 0
    coverage_weight = float(AppConfig.RETRIEVAL_MULTI_QUERY_RERANK_COVERAGE_WEIGHT)
    prior_weight = float(AppConfig.RETRIEVAL_MULTI_QUERY_RERANK_PRIOR_WEIGHT)
    route_rank_weight = float(AppConfig.RETRIEVAL_MULTI_QUERY_RERANK_ROUTE_RANK_WEIGHT)
    route_rank_window = max(1, int(AppConfig.RETRIEVAL_MULTI_QUERY_RERANK_ROUTE_RANK_WINDOW))
    if coverage_weight <= 0 and prior_weight <= 0 and route_rank_weight <= 0:
        return docs, 0

    boosted = 0
    blended: list[dict] = []
    for doc in docs:
        metadata = doc.get("metadata", {}) or {}
        coverage_count = int(metadata.get("__mq_coverage_count", 0) or 0)
        prior_score = float(metadata.get("__mq_prior_score", 0.0) or 0.0)
        best_rank = int(metadata.get("__mq_best_rank", 0) or 0)

        score = float(doc.get("score", 0.0))
        score += coverage_weight * float(max(0, coverage_count - 1))
        score += prior_weight * prior_score
        if best_rank > 0:
            score += route_rank_weight * max(0.0, (route_rank_window - best_rank + 1) / route_rank_window)

        if score > float(doc.get("score", 0.0)):
            doc["score"] = score
            boosted += 1
        blended.append(doc)
    blended.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return blended, boosted


def _rerank_route_candidates(
    route_candidates: list[dict],
    route_query: str,
    reranker: Reranker | None,
) -> list[dict]:
    if not route_candidates:
        return []
    if reranker is None:
        return list(route_candidates)
    input_results = _to_search_results(route_candidates)
    reranked = reranker.rerank(
        route_query,
        input_results,
        top_n=min(len(input_results), len(route_candidates)),
    )
    return _to_context_docs(reranked)


def _interleave_route_seed_docs(
    route_runs: list[dict],
    per_route_top_k: int,
) -> list[dict]:
    keep_k = max(1, int(per_route_top_k))
    buckets: list[list[dict]] = []
    for run in route_runs:
        docs = run.get("docs", []) or []
        if docs:
            buckets.append(list(docs[:keep_k]))
    if not buckets:
        return []

    out: list[dict] = []
    seen: set[str] = set()
    idx = 0
    while True:
        progressed = False
        for docs in buckets:
            if idx >= len(docs):
                continue
            progressed = True
            doc = docs[idx]
            doc_id = str(doc.get("id", "")).strip()
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(doc)
        if not progressed:
            break
        idx += 1
    return out


def rerank_phase1_node(state: AgentState) -> AgentState:
    query = state.get("retrieval_query") or state.get("query", "")
    retrieval_queries = _normalize_retrieval_queries(state=state, fallback_query=str(query or ""))
    multi_query_enabled = len(retrieval_queries) > 1
    rerank_query = str(query or "")
    if multi_query_enabled:
        rerank_query = " ; ".join(retrieval_queries)

    rerank_top_n = max(1, int(AppConfig.PHASE1_RERANK_TOP_N))
    candidates = state.get("phase1_candidates", [])[:rerank_top_n]
    phase1_query_routes = state.get("phase1_query_routes", []) or []
    reranker = _get_reranker()

    if reranker is None:
        reranked_docs = candidates
        rerank_enabled = False
    else:
        input_results = _to_search_results(candidates)
        reranked = reranker.rerank(
            rerank_query, input_results, top_n=min(rerank_top_n, len(input_results))
        )
        reranked_docs = _to_context_docs(reranked)
        rerank_enabled = True

    reranked_docs, coverage_boosted_docs = _blend_multi_query_coverage_score(
        reranked_docs,
        multi_query_enabled=multi_query_enabled,
    )
    reranked_docs, query_calibrated_docs = _apply_query_calibration(
        docs=reranked_docs,
        query=rerank_query,
    )
    reranked_docs, source_coverage_added = _enforce_source_coverage(
        ranked_docs=reranked_docs,
        candidates=candidates,
    )

    route_runs: list[dict] = []
    route_seed_docs: list[dict] = []
    if multi_query_enabled and len(phase1_query_routes) > 1:
        for route in phase1_query_routes:
            route_query = " ".join(str(route.get("query", "")).split()).strip()
            route_idx = int(route.get("route_index", len(route_runs)))
            route_candidates = list(route.get("candidates", []) or [])
            route_docs = _rerank_route_candidates(
                route_candidates=route_candidates,
                route_query=route_query or rerank_query,
                reranker=reranker,
            )
            route_docs, _ = _apply_query_calibration(
                docs=route_docs,
                query=route_query or rerank_query,
            )
            normalized_route_docs: list[dict] = []
            for rank, doc in enumerate(route_docs, start=1):
                metadata = dict(doc.get("metadata", {}) or {})
                metadata["__mq_route_index"] = route_idx
                metadata["__mq_route_query"] = route_query
                metadata["__mq_route_rank"] = rank
                doc["metadata"] = metadata
                normalized_route_docs.append(doc)
            route_runs.append(
                {
                    "route_index": route_idx,
                    "query": route_query,
                    "docs": normalized_route_docs,
                }
            )

        route_seed_docs = _interleave_route_seed_docs(
            route_runs=route_runs,
            per_route_top_k=int(AppConfig.RETRIEVAL_MULTI_QUERY_CONTEXT_TOP_K),
        )
        reranked_docs = _dedupe_docs_by_id(route_seed_docs + reranked_docs)

        # 仅保留轻量 route 数据给 compose，用于保证每个子 query 至少有证据进入提示。
        compact_routes: list[dict] = []
        keep_k = max(1, int(AppConfig.RETRIEVAL_MULTI_QUERY_CONTEXT_TOP_K))
        for run in route_runs:
            compact_routes.append(
                {
                    "route_index": run.get("route_index", 0),
                    "query": run.get("query", ""),
                    "candidate_count": len(run.get("docs", []) or []),
                    "candidates": list(run.get("docs", [])[:keep_k]),
                }
            )
        state["phase1_query_routes"] = compact_routes
    else:
        state["phase1_query_routes"] = []

    source_coverage_count = len({_extract_source_name(doc) for doc in reranked_docs})

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
                    "query_preview": clip_text(rerank_query, 180),
                    "retrieval_query_count": len(retrieval_queries),
                },
                "memory": {
                    "rerank_top_n": rerank_top_n,
                    "rerank_input_count": len(candidates),
                    "rerank_output_count": len(reranked_docs),
                    "rerank_enabled": rerank_enabled,
                    "coverage_boosted_docs": coverage_boosted_docs,
                    "query_calibrated_docs": query_calibrated_docs,
                    "query_route_count": len(route_runs),
                    "route_seed_docs": len(route_seed_docs),
                    "source_coverage_count": source_coverage_count,
                    "source_coverage_added": source_coverage_added,
                    "context_pool_preview": preview_docs(reranked_docs),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
