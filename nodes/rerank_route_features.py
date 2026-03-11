import re
from pathlib import Path

from config.settings import AppConfig

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Generic alias normalization for current filing scope; not tied to one query.
_STATIC_ENTITY_ALIAS_MAP: dict[str, str] = {
    "amazon": "amazon",
    "amzn": "amazon",
    "alphabet": "alphabet",
    "google": "alphabet",
    "googl": "alphabet",
    "goog": "alphabet",
    "microsoft": "microsoft",
    "msft": "microsoft",
}

_IGNORE_SOURCE_TOKENS = {
    "10",
    "10k",
    "k",
    "pdf",
    "form",
    "annual",
    "report",
    "inc",
    "incorporated",
}

_BROAD_CASH_SCOPE_TERMS = (
    "cash, cash equivalents, and short term marketable securities",
    "cash, cash equivalents, and marketable securities",
    "total cash, cash equivalents, and short term investments",
    "cash equivalents and marketable debt securities",
    "cash equivalents and marketable securities",
)

_EXACT_CASH_SCOPE_TERMS = (
    "cash and cash equivalents",
    "cash equivalents, end of period",
)


def _normalize_entity_token(token: str) -> str:
    t = str(token or "").strip().lower()
    if not t:
        return ""
    return _STATIC_ENTITY_ALIAS_MAP.get(t, t)


def _dynamic_source_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for src in AppConfig.RETRIEVAL_ALLOWED_SOURCES:
        basename = Path(str(src or "")).name.lower()
        if not basename:
            continue
        stem = basename.rsplit(".", 1)[0]
        parts = [p for p in _NON_ALNUM_RE.split(stem) if p]
        for part in parts:
            if part.isdigit():
                continue
            if part in _IGNORE_SOURCE_TOKENS or len(part) < 3:
                continue
            aliases.setdefault(part, _normalize_entity_token(part))
    return aliases


def _entity_alias_map() -> dict[str, str]:
    out = dict(_STATIC_ENTITY_ALIAS_MAP)
    out.update(_dynamic_source_aliases())
    return out


def entities_from_text(text: str) -> set[str]:
    lowered = str(text or "").lower()
    if not lowered:
        return set()
    alias_map = _entity_alias_map()
    found: set[str] = set()
    for tok in _TOKEN_RE.findall(lowered):
        canonical = alias_map.get(tok)
        if canonical:
            found.add(canonical)
    return found


def entities_from_doc(doc: dict) -> set[str]:
    metadata = doc.get("metadata", {}) or {}
    source_text = " ".join(
        [
            str(metadata.get("source", "") or ""),
            str(metadata.get("title", "") or ""),
            str(doc.get("title", "") or ""),
            str(doc.get("id", "") or ""),
            str(metadata.get("h1", "") or ""),
            str(metadata.get("h2", "") or ""),
        ]
    )
    # Keep content slice small to avoid overfitting to noisy body text.
    content_slice = str(doc.get("content", "") or "")[:260]
    return entities_from_text(" ".join([source_text, content_slice]))


def route_consistency_delta(query: str, doc: dict) -> float:
    query_lower = str(query or "").lower()
    if not query_lower.strip():
        return 0.0

    content_lower = str(doc.get("content", "") or "").lower()
    query_entities = entities_from_text(query_lower)
    doc_entities = entities_from_doc(doc)

    delta = 0.0

    # Entity consistency for route-specific queries.
    if len(query_entities) == 1 and doc_entities:
        entity_match_bonus = float(
            getattr(AppConfig, "RETRIEVAL_ROUTE_ENTITY_MATCH_BONUS", 0.10)
        )
        entity_mismatch_penalty = float(
            getattr(AppConfig, "RETRIEVAL_ROUTE_ENTITY_MISMATCH_PENALTY", 0.42)
        )
        if query_entities & doc_entities:
            delta += entity_match_bonus
        else:
            delta -= entity_mismatch_penalty

    # Metric scope consistency for cash-only queries.
    asks_cash_eq = (
        "cash and cash equivalents" in query_lower
        or "cash equivalents" in query_lower
    )
    if asks_cash_eq:
        exact_hit = any(term in content_lower for term in _EXACT_CASH_SCOPE_TERMS)
        broad_hit = any(term in content_lower for term in _BROAD_CASH_SCOPE_TERMS)
        metric_scope_penalty = float(
            getattr(AppConfig, "RETRIEVAL_ROUTE_METRIC_SCOPE_PENALTY", 0.16)
        )
        if broad_hit and not exact_hit:
            delta -= metric_scope_penalty
        elif exact_hit:
            delta += min(0.08, metric_scope_penalty * 0.5)

    return float(delta)
