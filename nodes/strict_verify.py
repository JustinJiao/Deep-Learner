import time
import re
from typing import Any
from pathlib import Path

from config.settings import AppConfig
from core.llm_call import run_prompt
from core.state import AgentState, StepLog
from llm.prompts.strict_verify import StrictVerifyPrompt
from nodes.log_utils import clip_text, preview_docs

ALLOWED_FAILURE_TYPES = {
    "INSUFFICIENT_EVIDENCE",
    "LOGICAL_ERROR",
    "CITATION_MISMATCH",
    "FORMAT_ERROR",
}

_TRIGGER_TO_FAILURE_TYPE = {
    "citation_missing": "CITATION_MISMATCH",
    "citation_fabricated": "CITATION_MISMATCH",
    "unsupported_claim": "LOGICAL_ERROR",
    "logic_contradiction": "LOGICAL_ERROR",
    "citation_score_too_low": "CITATION_MISMATCH",
    "hallucination_score_too_low": "INSUFFICIENT_EVIDENCE",
    "logic_score_too_low": "LOGICAL_ERROR",
    "total_score_below_threshold": "LOGICAL_ERROR",
    "insufficient_evidence_missing_company": "INSUFFICIENT_EVIDENCE",
}

_MISSING_EVIDENCE_PATTERN = re.compile(
    r"missing\s+explicit\s+evidence\s+for\s*:\s*([^\n\.]+)",
    flags=re.IGNORECASE,
)

_COMPANY_ALIAS_MAP: dict[str, str] = {
    "amazon": "Amazon",
    "microsoft": "Microsoft",
    "msft": "Microsoft",
    "alphabet": "Alphabet",
    "google": "Alphabet",
}

_MULTI_COMPANY_MARKERS = (
    "all three companies",
    "all three",
    "all companies",
    "these companies",
    "three companies",
)


def _clamp_score_0_5(value: object, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(5.0, score))


def _clamp_confidence(value: object, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return default


def _pick_verify_context(context_pool: list[dict]) -> list[dict]:
    top_k = int(AppConfig.RUNTIME_VERIFY_CONTEXT_TOP_K)
    if top_k <= 0:
        return list(context_pool)
    return list(context_pool[:top_k])


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_heading(value: object) -> str:
    text = str(value or "").strip()
    while text.startswith("#"):
        text = text[1:]
    return text.strip()


def _extract_source_and_module(doc: dict) -> tuple[str, str]:
    metadata = doc.get("metadata", {}) or {}

    source_raw = (
        metadata.get("source")
        or doc.get("source")
        or doc.get("title")
        or doc.get("id")
        or "Unknown Document"
    )
    source_text = str(source_raw).strip()
    source_name = Path(source_text).name or source_text or "Unknown Document"

    module = (
        _normalize_heading(metadata.get("h2"))
        or _normalize_heading(metadata.get("h1"))
        or _normalize_heading(doc.get("module"))
        or "General"
    )
    return source_name, module


def _build_verify_prompt_context(context_pool: list[dict]) -> list[dict]:
    prompt_docs: list[dict] = []
    for doc in context_pool:
        source_name, module = _extract_source_and_module(doc)
        doc_id = str(doc.get("id", "")).strip() or f"{source_name}::{module}"
        prompt_docs.append(
            {
                "id": doc_id,
                "title": source_name,
                "source": source_name,
                "module": module,
                "score": _safe_float(doc.get("score"), 0.0),
                "content": str(doc.get("content", "") or ""),
            }
        )
    return prompt_docs


def _context_chars(context_pool: list[dict]) -> int:
    chars = 0
    for doc in context_pool:
        chars += len(str(doc.get("content", "") or ""))
    return chars


def _normalize_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    citation = raw.get("citation", {}) or {}
    hallucination = raw.get("hallucination", {}) or {}
    logic = raw.get("logic", {}) or {}
    completeness = raw.get("completeness", {}) or {}
    fmt = raw.get("format", {}) or {}

    return {
        "citation": {
            "score": _clamp_score_0_5(citation.get("score"), default=0.0),
            "missing": _as_bool(citation.get("missing"), default=True),
            "fabricated": _as_bool(citation.get("fabricated"), default=False),
        },
        "hallucination": {
            "score": _clamp_score_0_5(hallucination.get("score"), default=0.0),
            "unsupported_claim": _as_bool(
                hallucination.get("unsupported_claim"),
                default=False,
            ),
        },
        "logic": {
            "score": _clamp_score_0_5(logic.get("score"), default=0.0),
            "contradiction": _as_bool(logic.get("contradiction"), default=False),
        },
        "completeness": {
            "score": _clamp_score_0_5(completeness.get("score"), default=0.0),
        },
        "format": {
            "score": _clamp_score_0_5(fmt.get("score"), default=0.0),
        },
        "confidence": _clamp_confidence(raw.get("confidence"), default=0.0),
    }


def _weighted_total_score(metrics: dict[str, Any]) -> float:
    w_citation = float(AppConfig.SV_WEIGHT_CITATION)
    w_hallucination = float(AppConfig.SV_WEIGHT_HALLUCINATION)
    w_logic = float(AppConfig.SV_WEIGHT_LOGIC)
    w_completeness = float(AppConfig.SV_WEIGHT_COMPLETENESS)
    w_format = float(AppConfig.SV_WEIGHT_FORMAT)
    w_sum = w_citation + w_hallucination + w_logic + w_completeness + w_format
    if w_sum <= 0:
        w_citation, w_hallucination, w_logic, w_completeness, w_format = (
            0.35,
            0.25,
            0.20,
            0.15,
            0.05,
        )
        w_sum = 1.0
    else:
        w_citation /= w_sum
        w_hallucination /= w_sum
        w_logic /= w_sum
        w_completeness /= w_sum
        w_format /= w_sum

    total = (
        metrics["citation"]["score"] * w_citation
        + metrics["hallucination"]["score"] * w_hallucination
        + metrics["logic"]["score"] * w_logic
        + metrics["completeness"]["score"] * w_completeness
        + metrics["format"]["score"] * w_format
    )
    return max(0.0, min(5.0, float(total)))


def _normalize_failure_type(value: object) -> str:
    text = str(value or "").strip().upper()
    if text not in ALLOWED_FAILURE_TYPES:
        return "FORMAT_ERROR"
    return text


def _extract_missing_entities(response_text: str) -> set[str]:
    text = str(response_text or "")
    m = _MISSING_EVIDENCE_PATTERN.search(text)
    if not m:
        return set()
    raw = m.group(1)
    items: set[str] = set()
    for part in raw.split(","):
        token = str(part or "").strip().strip(".")
        if token:
            items.add(token)
    return items


def _companies_from_text(text: str) -> set[str]:
    lowered = str(text or "").lower()
    found: set[str] = set()
    for alias, company in _COMPANY_ALIAS_MAP.items():
        if alias in lowered:
            found.add(company)
    return found


def _is_generic_multi_company_query(query: str) -> bool:
    lowered = str(query or "").lower()
    return any(marker in lowered for marker in _MULTI_COMPANY_MARKERS)


def _required_companies_for_query(query: str, context_pool: list[dict]) -> set[str]:
    project_default = {"Amazon", "Alphabet", "Microsoft"}
    query_companies = _companies_from_text(query)
    if len(query_companies) >= 2:
        return query_companies
    if not _is_generic_multi_company_query(query):
        return set()

    inferred: set[str] = set()
    for doc in context_pool:
        metadata = doc.get("metadata", {}) or {}
        source_text = " ".join(
            [
                str(metadata.get("source", "") or ""),
                str(doc.get("source", "") or ""),
                str(doc.get("title", "") or ""),
                str(doc.get("id", "") or ""),
            ]
        )
        inferred.update(_companies_from_text(source_text))
    if len(inferred) >= 2:
        return inferred
    return project_default


def _companies_from_citations(citations: list[dict]) -> set[str]:
    found: set[str] = set()
    for c in citations:
        text = " ".join(
            [
                str(c.get("id", "") or ""),
                str(c.get("title", "") or ""),
            ]
        )
        found.update(_companies_from_text(text))
    return found


def _missing_company_coverage(
    query: str,
    context_pool: list[dict],
    citations: list[dict],
) -> set[str]:
    required = _required_companies_for_query(query, context_pool)
    if len(required) < 2:
        return set()
    cited = _companies_from_citations(citations)
    return required - cited


def _decide_strict_action(
    metrics: dict[str, Any],
    total_score: float,
) -> tuple[str, str, str]:
    # 第一层：致命布尔拦截
    if bool(metrics["citation"]["missing"]):
        trigger = "citation_missing"
        return ("REPAIR", trigger, _normalize_failure_type(_TRIGGER_TO_FAILURE_TYPE[trigger]))
    if bool(metrics["citation"]["fabricated"]):
        trigger = "citation_fabricated"
        return ("REPAIR", trigger, _normalize_failure_type(_TRIGGER_TO_FAILURE_TYPE[trigger]))
    if bool(metrics["hallucination"]["unsupported_claim"]) and bool(
        AppConfig.SV_BLOCK_UNSUPPORTED_CLAIM
    ):
        trigger = "unsupported_claim"
        return ("REPAIR", trigger, _normalize_failure_type(_TRIGGER_TO_FAILURE_TYPE[trigger]))
    if bool(metrics["logic"]["contradiction"]):
        trigger = "logic_contradiction"
        return ("REPAIR", trigger, _normalize_failure_type(_TRIGGER_TO_FAILURE_TYPE[trigger]))

    # 第二层：单维度最低分拦截
    floor = float(AppConfig.SV_CRITICAL_SCORE_FLOOR)
    if float(metrics["citation"]["score"]) <= floor:
        trigger = "citation_score_too_low"
        return ("REPAIR", trigger, _normalize_failure_type(_TRIGGER_TO_FAILURE_TYPE[trigger]))
    if float(metrics["hallucination"]["score"]) <= floor:
        trigger = "hallucination_score_too_low"
        return ("REPAIR", trigger, _normalize_failure_type(_TRIGGER_TO_FAILURE_TYPE[trigger]))
    if float(metrics["logic"]["score"]) <= floor:
        trigger = "logic_score_too_low"
        return ("REPAIR", trigger, _normalize_failure_type(_TRIGGER_TO_FAILURE_TYPE[trigger]))

    # 第三层：加权总分
    total_threshold = float(AppConfig.SV_TOTAL_THRESHOLD)
    if total_score < total_threshold:
        trigger = "total_score_below_threshold"
        return ("REPAIR", trigger, _normalize_failure_type(_TRIGGER_TO_FAILURE_TYPE[trigger]))

    return ("PASS", "", "")


def strict_verify_node(state: AgentState) -> AgentState:
    full_context_pool = state.get("context_pool", []) or []
    verify_context_pool = _pick_verify_context(full_context_pool)
    verify_prompt_context_pool = _build_verify_prompt_context(verify_context_pool)
    prompt_state: AgentState = dict(state)
    prompt_state["context_pool"] = verify_prompt_context_pool
    out = run_prompt(StrictVerifyPrompt, prompt_state)

    metrics = _normalize_metrics(out)
    total_score = _weighted_total_score(metrics)
    missing_entities = _extract_missing_entities(str(state.get("response", "")))
    missing_company_coverage = _missing_company_coverage(
        query=str(state.get("query", "")),
        context_pool=full_context_pool,
        citations=state.get("citations", []) or [],
    )
    if missing_company_coverage:
        strict_action = "REPAIR"
        repair_trigger = "insufficient_evidence_missing_company"
        failure_type = "INSUFFICIENT_EVIDENCE"
    elif missing_entities:
        strict_action = "REPAIR"
        repair_trigger = "insufficient_evidence_missing_company"
        failure_type = "INSUFFICIENT_EVIDENCE"
    else:
        strict_action, repair_trigger, failure_type = _decide_strict_action(
            metrics,
            total_score,
        )

    if failure_type == "CITATION_MISMATCH" and (missing_entities or missing_company_coverage):
        failure_type = "INSUFFICIENT_EVIDENCE"
        repair_trigger = "insufficient_evidence_missing_company"

    verdict = "PASS" if strict_action == "PASS" else "FAIL"
    strict_score = max(0.0, min(1.0, total_score / 5.0))
    reason = (
        f"trigger={repair_trigger or 'none'}; total_score={total_score:.2f}; "
        f"citation={metrics['citation']['score']:.2f}, "
        f"hallucination={metrics['hallucination']['score']:.2f}, "
        f"logic={metrics['logic']['score']:.2f}, "
        f"completeness={metrics['completeness']['score']:.2f}, "
        f"format={metrics['format']['score']:.2f}"
    )
    if missing_entities:
        reason += f"; missing_entities={', '.join(sorted(missing_entities))}"
    if missing_company_coverage:
        reason += f"; missing_company_coverage={', '.join(sorted(missing_company_coverage))}"

    state["strict_verdict"] = verdict
    state["strict_score"] = strict_score
    state["strict_total_score"] = total_score
    state["strict_action"] = strict_action
    state["strict_metrics"] = metrics
    state["strict_confidence"] = float(metrics.get("confidence", 0.0))
    state["repair_trigger"] = repair_trigger
    state["strict_reason"] = reason
    state["verified_revision"] = state.get("response_revision", 0)
    if verdict == "FAIL":
        state["failure_type"] = failure_type
    else:
        state.pop("failure_type", None)
        state["repair_trigger"] = ""

    state.setdefault("steps_log", []).append(
        StepLog(
            node="strict_verify",
            info={
                "state": {
                    "response_preview": clip_text(state.get("response", ""), 180),
                    "context_pool_count": len(full_context_pool),
                    "verify_context_count": len(verify_context_pool),
                    "verify_context_chars": _context_chars(verify_context_pool),
                    "verify_prompt_context_chars": _context_chars(verify_prompt_context_pool),
                    "response_revision": state.get("response_revision", 0),
                    "verified_revision": state.get("verified_revision", 0),
                },
                "llm_input": {
                    "context_pool_preview": preview_docs(verify_prompt_context_pool),
                },
                "llm_output": {
                    "metrics": metrics,
                    "verdict": verdict,
                    "action": strict_action,
                    "total_score": total_score,
                    "score": strict_score,
                    "score_threshold": float(AppConfig.SV_TOTAL_THRESHOLD),
                    "repair_trigger": repair_trigger,
                    "failure_type": failure_type if verdict == "FAIL" else "",
                    "reason_preview": clip_text(reason, 220),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
