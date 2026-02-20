# nodes/verify.py

import time
from core.state import AgentState, StepLog
from core.llm_call import run_prompt
from llm.prompts.verify import VerifyPrompt
from nodes.log_utils import clip_text, preview_docs


def verify_node(state: AgentState) -> AgentState:
    out = run_prompt(VerifyPrompt, state)

    # ===== 解析返回 =====
    is_hallucination = bool(out.get("is_hallucination", False))
    verify_type = out.get("type")
    next_action = out.get("next_step")
    reason = out.get("reason")

    state["is_hallucination"] = is_hallucination
    state["critique"] = out

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
                    "error_type": verify_type or out.get("error_type"),
                    "next_step": next_action,
                    "critique_preview": clip_text(reason or out.get("critique", ""), 180),
                },
            },
            timestamp=time.time(),
        )
    )

    return state
