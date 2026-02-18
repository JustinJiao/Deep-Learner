# nodes/stm_write.py

from session.store import get_session, save_session
from memory.stm import STM
from core.state import AgentState, StepLog


def stm_write_node(state: AgentState) -> AgentState:
    ctx = get_session(state["session_id"])

    if not isinstance(ctx.stm, dict):
        raise TypeError("ctx.stm must be dict.")

    stm = STM(ctx.stm)

    query = state.get("query")
    response = state.get("response")

    if query and response:
        stm.append_turn(query, response)

    # 更新 recent window
    stm.update_recent_messages(window_size=3)

    save_session(state["session_id"], ctx)

    state.setdefault("steps_log", []).append(
        StepLog(node="stm_write", info="stored turn")
    )

    return state
