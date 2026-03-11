import re
import time
from pathlib import Path
from typing import Any

from core.state import AgentState, StepLog
from nodes.log_utils import clip_text, preview_docs
from nodes.rerank_route_features import entities_from_doc, entities_from_text, route_consistency_delta

_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_NUMBER_RE = re.compile(r"\$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?|\$?\s*\d+(?:\.\d+)?")
_PERIOD_END_RE = re.compile(
    r"(?:year\s+ended|ended)\s+([A-Za-z]+\s+\d{1,2},\s*20\d{2})",
    flags=re.IGNORECASE,
)


def _source_name(doc: dict) -> str:
    metadata = doc.get("metadata", {}) or {}
    source_raw = (
        metadata.get("source")
        or metadata.get("title")
        or doc.get("source")
        or doc.get("title")
        or doc.get("id")
        or "Unknown Document"
    )
    source_text = str(source_raw).strip()
    return Path(source_text).name or source_text or "Unknown Document"


def _target_year(*texts: str) -> int | None:
    years: list[int] = []
    for text in texts:
        years.extend(int(y) for y in _YEAR_RE.findall(str(text or "")))
    if not years:
        return None
    return max(years)


def _infer_metric(route_query: str, original_query: str) -> str:
    q = f"{route_query} {original_query}".lower()
    if "cash and cash equivalents" in q or "cash equivalents" in q:
        return "cash_and_cash_equivalents"
    if "revenue" in q or "net sales" in q or "sales" in q:
        return "revenue"
    if "employee" in q or "headcount" in q:
        return "employees"
    if "risk" in q or "challenge" in q:
        return "risk"
    return "generic"


def _is_numeric_intent(route_query: str, original_query: str, metric: str) -> bool:
    q = f"{route_query} {original_query}".lower()
    if metric in {"cash_and_cash_equivalents", "revenue", "employees"}:
        return True
    numeric_terms = (
        "how much",
        "largest",
        "highest",
        "lowest",
        "compare",
        "increase",
        "decrease",
        "growth",
        "percent",
        "%",
        "number of",
    )
    return any(term in q for term in numeric_terms)


def _numeric_tokens(text: str) -> list[str]:
    src = str(text or "")
    tokens: list[str] = []
    for m in _NUMBER_RE.finditer(src):
        token = str(m.group(0) or "").strip()
        if not token:
            continue
        end = int(m.end())
        if end < len(src) and src[end:end + 1] == "%":
            continue
        normalized = token.replace("$", "").replace(",", "").strip()
        try:
            value = float(normalized)
        except ValueError:
            continue
        # Skip year-like numbers.
        if 1900.0 <= value <= 2100.0 and "," not in token and "." not in token:
            continue
        tokens.append(token)
    return tokens


def _token_to_million(token: str, context_window: str) -> float | None:
    normalized = str(token or "").replace("$", "").replace(",", "").strip()
    if not normalized:
        return None
    try:
        value = float(normalized)
    except ValueError:
        return None
    lowered = str(context_window or "").lower()
    if "billion" in lowered:
        return value * 1000.0
    return value


def _metric_anchor_score(metric: str, content_lower: str) -> float:
    if metric == "cash_and_cash_equivalents":
        score = 0.0
        if "cash and cash equivalents" in content_lower:
            score += 0.8
        elif "cash equivalents" in content_lower:
            score += 0.45
        if "cash, cash equivalents, and marketable securities" in content_lower:
            score -= 0.18
        if "cash, cash equivalents, and short term marketable securities" in content_lower:
            score -= 0.20
        if "total cash, cash equivalents, and short term investments" in content_lower:
            score -= 0.18
        return score
    if metric == "revenue":
        if "revenue" in content_lower or "net sales" in content_lower or "sales" in content_lower:
            return 0.7
        return -0.2
    if metric == "employees":
        if "employees" in content_lower or "headcount" in content_lower:
            return 0.6
        return -0.1
    if metric == "risk":
        if "risk" in content_lower or "regulation" in content_lower or "uncertaint" in content_lower:
            return 0.5
        return -0.1
    return 0.0


def _candidate_fact_score(
    route_query: str,
    doc: dict,
    target_entity: str,
    metric: str,
    target_year: int | None,
) -> float:
    content = str(doc.get("content", "") or "")
    content_lower = content.lower()
    score = float(doc.get("score", 0.0) or 0.0) * 0.16

    doc_entities = entities_from_doc(doc)
    if target_entity:
        if doc_entities and target_entity in doc_entities:
            score += 0.80
        elif doc_entities and target_entity not in doc_entities:
            score -= 0.80

    score += _metric_anchor_score(metric=metric, content_lower=content_lower)

    if target_year is not None and str(target_year) in content:
        score += 0.30

    if _numeric_tokens(content):
        score += 0.22

    score += route_consistency_delta(route_query, doc)
    return score


def _extract_best_quote(
    route_query: str,
    content: str,
    target_entity: str,
    metric: str,
    target_year: int | None,
) -> str:
    src = str(content or "")
    if not src.strip():
        return ""

    blocks = [b.strip() for b in re.split(r"\n{2,}", src) if b and b.strip()]
    if not blocks:
        blocks = [src.strip()]

    best = blocks[0]
    best_score = -1e9
    for blk in blocks[:120]:
        blk_lower = blk.lower()
        s = 0.0
        s += _metric_anchor_score(metric=metric, content_lower=blk_lower) * 2.0
        if target_year is not None and str(target_year) in blk:
            s += 0.8
        if target_entity and target_entity in entities_from_text(blk_lower):
            s += 0.6
        nums = _numeric_tokens(blk)
        s += min(0.8, 0.25 * len(nums))
        if "table p" in blk_lower:
            s += 0.25
        if s > best_score:
            best_score = s
            best = blk

    clipped = " ".join(best.split())
    if len(clipped) > 260:
        clipped = clipped[:260]
    return clipped


def _extract_period_end(text: str) -> str:
    m = _PERIOD_END_RE.search(str(text or ""))
    if not m:
        return ""
    return " ".join(str(m.group(1) or "").split()).strip()


def _extract_value_from_quote(quote: str) -> tuple[str, float | None]:
    tokens = _numeric_tokens(quote)
    if not tokens:
        return "", None

    ranked: list[tuple[str, float | None]] = []
    for token in tokens:
        idx = str(quote).find(token)
        win = str(quote)[max(0, idx - 24): min(len(str(quote)), idx + len(token) + 24)] if idx >= 0 else str(quote)
        million = _token_to_million(token, context_window=win)
        ranked.append((token, million))

    # Prefer material financial magnitudes over day/month numbers.
    strong = [
        (tok, val)
        for tok, val in ranked
        if ("," in tok) or (val is not None and val >= 1000.0)
    ]
    chosen = strong[-1] if strong else ranked[-1]
    return chosen


def _route_records(state: AgentState) -> list[dict]:
    routes = list(state.get("phase1_query_routes", []) or [])
    if routes:
        return routes
    retrieval_queries = list(state.get("retrieval_queries", []) or [])
    synthesized: list[dict] = []
    for idx, q in enumerate(retrieval_queries):
        synthesized.append(
            {
                "route_index": idx,
                "query": q,
                "candidate_count": 0,
                "candidates": [],
            }
        )
    return synthesized


def _fact_from_route(
    route: dict,
    state: AgentState,
) -> dict[str, Any]:
    original_query = str(state.get("query", "") or "")
    resolved_query = str(state.get("resolved_query", "") or "")
    route_query = " ".join(str(route.get("query", "")).split()).strip()
    route_idx = int(route.get("route_index", 0) or 0)
    target_year = _target_year(route_query, resolved_query, original_query)
    metric = _infer_metric(route_query=route_query, original_query=original_query)
    numeric_intent = _is_numeric_intent(route_query, original_query, metric)

    query_entities = entities_from_text(route_query)
    target_entity = next(iter(sorted(query_entities))) if query_entities else ""

    candidates = list(route.get("candidates", []) or [])
    if not candidates:
        candidates = list(state.get("context_pool", []) or [])[:24]

    best_doc: dict | None = None
    best_score = -1e9
    for cand in candidates:
        score = _candidate_fact_score(
            route_query=route_query,
            doc=cand,
            target_entity=target_entity,
            metric=metric,
            target_year=target_year,
        )
        if score > best_score:
            best_score = score
            best_doc = cand

    if not best_doc:
        return {
            "route_index": route_idx,
            "route_query": route_query,
            "entity": target_entity,
            "metric": metric,
            "target_year": target_year,
            "missing": True,
            "missing_reason": "no_candidates",
            "confidence": 0.0,
        }

    quote = _extract_best_quote(
        route_query=route_query,
        content=str(best_doc.get("content", "") or ""),
        target_entity=target_entity,
        metric=metric,
        target_year=target_year,
    )
    value_text, value_million = _extract_value_from_quote(quote)
    period_end = _extract_period_end(quote)
    source_name = _source_name(best_doc)

    missing = False
    missing_reason = ""
    if numeric_intent and not value_text:
        missing = True
        missing_reason = "numeric_value_not_found_in_best_evidence"
    elif target_entity:
        doc_entities = entities_from_doc(best_doc)
        if doc_entities and target_entity not in doc_entities:
            missing = True
            missing_reason = "best_evidence_entity_mismatch"

    confidence = max(0.0, min(1.0, (best_score + 1.5) / 4.0))
    return {
        "route_index": route_idx,
        "route_query": route_query,
        "entity": target_entity,
        "metric": metric,
        "target_year": target_year,
        "citation_id": str(best_doc.get("id", "")).strip(),
        "citation_title": source_name,
        "quote": quote,
        "value_text": value_text,
        "value_million": value_million,
        "fiscal_period_end": period_end,
        "missing": missing,
        "missing_reason": missing_reason,
        "confidence": confidence,
    }


def extract_route_facts_node(state: AgentState) -> AgentState:
    routes = _route_records(state)
    route_facts = [_fact_from_route(route=route, state=state) for route in routes]

    missing_routes = [int(f.get("route_index", -1)) for f in route_facts if bool(f.get("missing", False))]
    covered_routes = [int(f.get("route_index", -1)) for f in route_facts if not bool(f.get("missing", False))]
    missing_entities = sorted(
        {
            str(f.get("entity", "")).strip()
            for f in route_facts
            if bool(f.get("missing", False)) and str(f.get("entity", "")).strip()
        }
    )

    coverage = {
        "route_count": len(route_facts),
        "covered_route_count": len(covered_routes),
        "missing_route_count": len(missing_routes),
        "missing_routes": missing_routes,
        "missing_entities": missing_entities,
    }

    state["route_facts"] = route_facts
    state["route_fact_coverage"] = coverage

    state.setdefault("steps_log", []).append(
        StepLog(
            node="extract_route_facts",
            info={
                "state": {
                    "query_preview": clip_text(state.get("query", ""), 180),
                    "resolved_query_preview": clip_text(state.get("resolved_query", ""), 180),
                    "route_count": len(routes),
                    "context_pool_count": len(state.get("context_pool", []) or []),
                },
                "memory": {
                    "coverage": coverage,
                    "route_facts_preview": route_facts[:6],
                    "context_pool_preview": preview_docs((state.get("context_pool", []) or [])[:10]),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
