# nodes/finalize.py
import time
from core.state import AgentState, StepLog


def finalize_node(state: AgentState) -> AgentState:
    state.setdefault("steps_log", []).append(
        StepLog(
            node="finalize",
            info="workflow finished",
            timestamp=time.time(),
        )
    )
    return state
