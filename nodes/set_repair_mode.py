import time

from core.state import AgentState, StepLog
from nodes.log_utils import clip_text


def set_repair_mode_node(state: AgentState) -> AgentState:
    reason = str(state.get("strict_reason", "")).strip()
    failure_type = str(state.get("failure_type", "")).strip()
    repair_trigger = str(state.get("repair_trigger", "")).strip()

    state["repair_mode"] = True
    state["repair_used"] = True
    if repair_trigger:
        state["repair_reason"] = f"{repair_trigger} | {failure_type}: {reason}".strip(": ")
    else:
        state["repair_reason"] = f"{failure_type}: {reason}".strip(": ")

    state.setdefault("steps_log", []).append(
        StepLog(
            node="set_repair_mode",
            info={
                "state": {
                    "repair_mode": True,
                    "repair_used": True,
                    "repair_trigger": repair_trigger,
                    "failure_type": failure_type,
                    "repair_reason_preview": clip_text(state.get("repair_reason", ""), 220),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
