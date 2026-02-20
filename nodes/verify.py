# nodes/verify.py

import time
from core.state import AgentState, StepLog
from core.llm_call import run_prompt
from llm.prompts.verify import VerifyPrompt
from nodes.log_utils import clip_text, preview_docs


def _normalize_verify_output(out: dict) -> dict:
    is_hallucination = bool(out.get("is_hallucination", False))
    critique = out.get("critique") or out.get("reason") or ""

    if not is_hallucination:
        return {
            "is_hallucination": False,
            "critique": critique,
            "error_type": "",
            "next_step": "",
        }

    error_type = out.get("error_type") or out.get("type") or "generation_error"
    next_step = out.get("next_step") or "compose"
    if next_step not in {"compose", "retrieve", "query_rewrite"}:
        next_step = "compose"

    return {
        "is_hallucination": True,
        "critique": critique,
        "error_type": error_type,
        "next_step": next_step,
    }


def verify_node(state: AgentState) -> AgentState:
    out = run_prompt(VerifyPrompt, state)
    normalized = _normalize_verify_output(out)

    # ===== 解析返回 =====
    is_hallucination = normalized["is_hallucination"]
    verify_type = normalized["error_type"]
    next_action = normalized["next_step"]
    critique = normalized["critique"]

    state["is_hallucination"] = is_hallucination
    state["critique"] = normalized

    # PASS 时清掉旧的 repair_hint
    if not is_hallucination:
        state.pop("repair_hint", None)

    verdict = "FAIL" if is_hallucination else "PASS"

    # ===== 增强调试日志 =====
    state.setdefault("steps_log", []).append(
        StepLog(
            node="verify",
            info={
                "state": {
                    "response_preview": clip_text(state.get("response", ""), 180),
                    "context_pool_count": len(state.get("context_pool", [])),
                },
                "llm_input": {
                    "response_preview": clip_text(state.get("response", ""), 180),
                    "context_pool_preview": preview_docs(state.get("context_pool", [])),
                },
                "llm_output": {
                    "verdict": verdict,
                    "error_type": verify_type,
                    "next_step": next_action,
                    "critique_preview": clip_text(critique, 180),
                },
            },
            timestamp=time.time(),
        )
    )

    return state
