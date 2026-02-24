import time

from core.state import AgentState, StepLog
from nodes.log_utils import clip_text


def degrade_or_abstain_node(state: AgentState) -> AgentState:
    reason = state.get("strict_reason", "") or state.get("memory_reason", "")
    failure_type = state.get("failure_type", "")
    answer = (
        "当前证据不足或存在冲突，我无法给出可验证的结论。\n"
        "为避免误导，本次先给出保守回答：不确定。"
    )
    if reason:
        answer += f"\n原因：{reason}"
    if failure_type:
        answer += f"\n失败类型：{failure_type}"

    state["response"] = answer
    state["citations"] = []
    state["run_status"] = "degraded"

    state.setdefault("steps_log", []).append(
        StepLog(
            node="degrade_or_abstain",
            info={
                "state": {
                    "strict_reason_preview": clip_text(reason, 220),
                    "failure_type": failure_type,
                    "response_preview": clip_text(answer, 220),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
