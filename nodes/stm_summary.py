# nodes/stm_summary.py

from core.state import AgentState, StepLog
from memory.stm import STM
from session.store import get_session, save_session
from core.llm_call import run_prompt
from llm.prompts.stm_compress import STMCompressPrompt


def stm_summary_node(state: AgentState) -> AgentState:
    ctx = get_session(state["session_id"])
    stm = STM(ctx.stm)

    # 阈值可配置（默认 5）
    threshold = 5

    # 这里用 STM 的真实指针判断是否需要压缩（不要依赖 stm_write 的 bool 标记）
    if not stm.need_compress(threshold=threshold):
        return state

    chunk = stm.get_chunk_to_compress(threshold=threshold)

    # turn -> 可读文本（供 LLM 压缩）
    lines = []
    for t in chunk:
        q = (t.get("query") or "").strip()
        r = (t.get("response") or "").strip()
        if q:
            lines.append(f"用户: {q}")
        if r:
            lines.append(f"助手: {r}")
        lines.append("")  # blank line between turns

    # ★关键：字段名要和 STMCompressPrompt.READs 一致
    state["compress_chunk_text"] = "\n".join(lines).strip()

    out = run_prompt(STMCompressPrompt, state)

    compressed_text = (out.get("stm_compressed_text") or "").strip()
    if compressed_text:
        stm.append_summary(compressed_text)
        stm.mark_compressed(threshold=threshold)

        # 保存 session
        save_session(state["session_id"], ctx)

        state.setdefault("steps_log", []).append(
            StepLog(node="stm_summary", info=f"summary merged until={stm.compressed_until}")
        )

    return state
