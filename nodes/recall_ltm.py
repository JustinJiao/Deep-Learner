# nodes/recall_ltm.py
from core.state import AgentState, StepLog
from memory.ltm import LTM


def recall_ltm_node(state: AgentState) -> AgentState:
    ltm = LTM()
    memories = ltm.recall(state["query"])

    state["long_term_memory"] = " | ".join(memories) if memories else "无相关长期记忆"

    state.setdefault("steps_log", []).append(
        StepLog(node="recall_ltm", info=f"memories={len(memories)}")
    )
    return state
