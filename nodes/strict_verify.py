import time

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

_UNCERTAIN_MARKERS = (
    "不确定",
    "无法确定",
    "证据不足",
    "uncertain",
    "cannot determine",
    "insufficient evidence",
    "not sure",
)


def _normalize_verdict(value: object) -> str:
    text = str(value or "").strip().upper()
    if text not in {"PASS", "FAIL"}:
        return "FAIL"
    return text


def _clamp_score(value: object, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, score))


def _normalize_failure_type(value: object) -> str:
    text = str(value or "").strip().upper()
    if text not in ALLOWED_FAILURE_TYPES:
        return "FORMAT_ERROR"
    return text


def _is_uncertain_response(text: object) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    lowered = raw.lower()
    return any(marker in raw or marker in lowered for marker in _UNCERTAIN_MARKERS)


def _pick_verify_context(context_pool: list[dict]) -> list[dict]:
    top_k = int(getattr(AppConfig, "RUNTIME_VERIFY_CONTEXT_TOP_K", 8))
    if top_k <= 0:
        return list(context_pool)
    return list(context_pool[:top_k])


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_verify_prompt_context(context_pool: list[dict]) -> list[dict]:
    prompt_docs: list[dict] = []
    for doc in context_pool:
        prompt_docs.append(
            {
                "id": str(doc.get("id", "")).strip(),
                "title": str(doc.get("title", "")).strip(),
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


def strict_verify_node(state: AgentState) -> AgentState:
    full_context_pool = state.get("context_pool", []) or []
    verify_context_pool = _pick_verify_context(full_context_pool)
    verify_prompt_context_pool = _build_verify_prompt_context(verify_context_pool)
    prompt_state: AgentState = dict(state)
    prompt_state["context_pool"] = verify_prompt_context_pool
    out = run_prompt(StrictVerifyPrompt, prompt_state)

    llm_verdict = _normalize_verdict(out.get("verdict"))
    threshold = float(getattr(AppConfig, "STRICT_VERIFY_PASS_SCORE_THRESHOLD", 0.75))
    default_score = 0.9 if llm_verdict == "PASS" else 0.2
    strict_score = _clamp_score(out.get("score"), default=default_score)
    verdict = "PASS" if strict_score >= threshold else "FAIL"
    failure_type = _normalize_failure_type(out.get("failure_type"))
    reason = str(out.get("reason", "")).strip()

    # 当已有检索证据但回答仍为“不确定”时，将失败类型映射为逻辑错误，引导进入 repair。
    if (
        verdict == "FAIL"
        and failure_type == "INSUFFICIENT_EVIDENCE"
        and verify_context_pool
        and _is_uncertain_response(state.get("response", ""))
    ):
        failure_type = "LOGICAL_ERROR"
        if reason:
            reason += " | auto-remap: uncertain response despite retrieved evidence"
        else:
            reason = "auto-remap: uncertain response despite retrieved evidence"

    state["strict_verdict"] = verdict
    state["strict_score"] = strict_score
    state["strict_reason"] = reason
    state["verified_revision"] = state.get("response_revision", 0)
    if verdict == "FAIL":
        state["failure_type"] = failure_type
    else:
        state.pop("failure_type", None)

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
                    "verdict": verdict,
                    "llm_verdict": llm_verdict,
                    "score": strict_score,
                    "score_threshold": threshold,
                    "failure_type": failure_type if verdict == "FAIL" else "",
                    "reason_preview": clip_text(reason, 220),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
