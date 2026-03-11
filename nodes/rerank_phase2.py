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

    reranked_docs, query_calibrated_docs = _apply_query_calibration(
        docs=reranked_docs,
        query=str(query or ""),
    )
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
                    "query_calibrated_docs": query_calibrated_docs,
                    "source_coverage_count": source_coverage_count,
                    "source_coverage_added": source_coverage_added,
                    "context_pool_preview": preview_docs(reranked_docs),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
