import time

from core.state import AgentState, StepLog
from nodes.log_utils import clip_text


def set_repair_mode_node(state: AgentState) -> AgentState:
    reason = str(state.get("strict_reason", "")).strip()
    failure_type = str(state.get("failure_type", "")).strip()

    state["repair_mode"] = True
    state["repair_used"] = True
    state["repair_reason"] = f"{failure_type}: {reason}".strip(": ")

    state.setdefault("steps_log", []).append(
        StepLog(
            node="set_repair_mode",
            info={
                "state": {
                    "repair_mode": True,
                    "repair_used": True,
                    "failure_type": failure_type,
                    "repair_reason_preview": clip_text(state.get("repair_reason", ""), 220),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
