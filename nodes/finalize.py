# nodes/finalize.py
import time
from core.state import AgentState, StepLog
from nodes.log_utils import clip_text


def finalize_node(state: AgentState) -> AgentState:
    state.setdefault("steps_log", []).append(
        StepLog(
            node="finalize",
            info={
                "state": {
                    "run_status": state.get("run_status"),
                    "loop_count": state.get("loop_count"),
                    "is_hallucination": state.get("is_hallucination"),
                    "response_preview": clip_text(state.get("response", ""), 180),
                    "citations_count": len(state.get("citations", [])),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
