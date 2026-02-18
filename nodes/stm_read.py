# nodes/stm_read.py

from session.store import get_session
from memory.stm import STM
from core.state import AgentState, StepLog


def stm_read_node(state: AgentState) -> AgentState:
    ctx = get_session(state["session_id"])

    if not isinstance(ctx.stm, dict):
        raise TypeError("ctx.stm must be dict.")

    stm = STM(ctx.stm)

    # --------------------------
    # summary 给 LLM
    # --------------------------
    state["short_term_memory"] = stm.get_summary_text()

    # --------------------------
    # turn -> message 转换
    # --------------------------
    messages_for_llm = []

    for turn in stm.recent_messages:
        if turn.get("query"):
            messages_for_llm.append({
                "role": "user",
                "content": turn["query"]
            })
        if turn.get("response"):
            messages_for_llm.append({
                "role": "assistant",
                "content": turn["response"]
            })

    state["recent_messages"] = messages_for_llm

    state.setdefault("steps_log", []).append(
        StepLog(node="stm_read", info="mapped stm -> state")
    )

    return state
