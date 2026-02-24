import time

from config.settings import AppConfig
from core.llm_call import run_prompt
from core.state import AgentState, StepLog
from llm.prompts.verify_memory import VerifyMemoryPrompt
from nodes.log_utils import clip_text

_UNCERTAIN_MARKERS = (
    "不确定",
    "无法确定",
    "证据不足",
    "需检索",
    "uncertain",
    "cannot determine",
    "insufficient evidence",
    "need retrieve",
)


def _normalize_verdict(value: object) -> str:
    text = str(value or "").strip().upper()
    if text not in {"SUFFICIENT", "NEED_RETRIEVE"}:
        return "NEED_RETRIEVE"
    return text


def _normalize_risk(value: object) -> str:
    text = str(value or "").strip().upper()
    if text not in {"LOW", "MEDIUM", "HIGH"}:
        return "MEDIUM"
    return text


def _clamp_score(value: object, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def _risk_from_score(score: float, threshold: float) -> str:
    if score >= max(0.88, threshold + 0.12):
        return "LOW"
    if score >= threshold:
        return "MEDIUM"
    return "HIGH"


def _is_uncertain_text(value: object) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return True
    lowered = raw.lower()
    return any(marker in raw or marker in lowered for marker in _UNCERTAIN_MARKERS)


def verify_memory_node(state: AgentState) -> AgentState:
    ltm_hits_count = int(state.get("ltm_hits_count", 0) or 0)
    used_memory_chunks = int(state.get("used_memory_chunks", 0) or 0)
    draft_confidence = float(state.get("draft_confidence", 0.0) or 0.0)
    draft_answer = str(state.get("draft_answer", "") or "")
    effective_query = str(state.get("resolved_query") or state.get("query", "")).strip()
    prompt_state: AgentState = dict(state)
    prompt_state["query"] = effective_query
    threshold = float(getattr(AppConfig, "MEMORY_SUFFICIENT_SCORE_THRESHOLD", 0.70))

    shortcut_no_memory = (ltm_hits_count <= 0 and used_memory_chunks <= 0)
    shortcut_low_conf = (
        used_memory_chunks <= 0
        and (draft_confidence < 0.55 or _is_uncertain_text(draft_answer))
    )
    model_risk_level = "MEDIUM"

    if shortcut_no_memory:
        memory_score = 0.0
        reason = "no memory evidence: ltm_hits_count=0 and used_memory_chunks=0"
        model_risk_level = "HIGH"
    elif shortcut_low_conf:
        memory_score = min(draft_confidence, 0.45)
        reason = "memory draft is low confidence or uncertain"
        model_risk_level = "HIGH"
    else:
        out = run_prompt(VerifyMemoryPrompt, prompt_state)
        llm_verdict = _normalize_verdict(out.get("verdict"))
        default_score = 0.85 if llm_verdict == "SUFFICIENT" else 0.35
        memory_score = _clamp_score(out.get("score"), default=default_score)
        reason = str(out.get("reason", "")).strip()
        model_risk_level = _normalize_risk(out.get("risk_level"))

    verdict = "SUFFICIENT" if memory_score >= threshold else "NEED_RETRIEVE"
    risk_level = _risk_from_score(memory_score, threshold)
    risk_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    if risk_rank[model_risk_level] > risk_rank[risk_level]:
        risk_level = model_risk_level
    if verdict == "NEED_RETRIEVE" and risk_level != "HIGH":
        risk_level = "HIGH"

    state["memory_score"] = memory_score
    state["memory_verdict"] = verdict
    state["memory_reason"] = reason
    state["memory_risk_level"] = risk_level

    state.setdefault("steps_log", []).append(
        StepLog(
            node="verify_memory",
            info={
                "state": {
                    "query_preview": clip_text(state.get("query", ""), 180),
                    "effective_query_preview": clip_text(effective_query, 180),
                    "draft_confidence": state.get("draft_confidence", 0.0),
                    "used_memory_chunks": state.get("used_memory_chunks", 0),
                    "ltm_hits_count": ltm_hits_count,
                },
                "llm_output": {
                    "score": memory_score,
                    "score_threshold": threshold,
                    "verdict": verdict,
                    "risk_level": risk_level,
                    "reason_preview": clip_text(reason, 220),
                    "shortcut_no_memory": shortcut_no_memory,
                    "shortcut_low_conf": shortcut_low_conf,
                },
            },
            timestamp=time.time(),
        )
    )
    return state
