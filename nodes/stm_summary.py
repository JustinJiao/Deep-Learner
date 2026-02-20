# nodes/stm_summary.py

import time

from core.state import AgentState, StepLog
from memory.stm import STM
from session.store import get_session, save_session
from core.llm_call import run_prompt
from llm.prompts.stm_compress import STMCompressPrompt
from nodes.log_utils import clip_text, preview_turns


def stm_summary_node(state: AgentState) -> AgentState:
    ctx = get_session(state["session_id"])
    stm = STM(ctx.stm)

    # 阈值可配置（默认 5）
    threshold = 5

    # 这里用 STM 的真实指针判断是否需要压缩（不要依赖 stm_write 的 bool 标记）
    if not stm.need_compress(threshold=threshold):
        state.setdefault("steps_log", []).append(
            StepLog(
                node="stm_summary",
                info={
                    "memory": {
                        "need_compress": False,
                        "messages_count": len(stm.messages),
                        "summary_blocks": len(stm.summary),
                        "compressed_until": stm.compressed_until,
                    },
                },
                timestamp=time.time(),
            )
        )
        return state

    chunk = stm.get_chunk_to_compress(threshold=threshold)
    compressed_until_before = stm.compressed_until
    summary_blocks_before = len(stm.summary)

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
            StepLog(
                node="stm_summary",
                info={
                    "memory": {
                        "need_compress": True,
                        "chunk_preview": preview_turns(chunk),
                        "compressed_until_before": compressed_until_before,
                        "compressed_until_after": stm.compressed_until,
                        "summary_blocks_before": summary_blocks_before,
                        "summary_blocks_after": len(stm.summary),
                    },
                    "llm_input": {
                        "compress_chunk_text_preview": clip_text(state.get("compress_chunk_text", ""), 180),
                    },
                    "llm_output": {
                        "stm_compressed_text_preview": clip_text(compressed_text, 180),
                    },
                },
                timestamp=time.time(),
            )
        )
    else:
        state.setdefault("steps_log", []).append(
            StepLog(
                node="stm_summary",
                info={
                    "memory": {
                        "need_compress": True,
                        "chunk_preview": preview_turns(chunk),
                        "compressed_until_before": compressed_until_before,
                        "compressed_until_after": stm.compressed_until,
                        "summary_blocks_before": summary_blocks_before,
                        "summary_blocks_after": len(stm.summary),
                    },
                    "llm_input": {
                        "compress_chunk_text_preview": clip_text(state.get("compress_chunk_text", ""), 180),
                    },
                    "llm_output": {
                        "stm_compressed_text_preview": "",
                    },
                },
                timestamp=time.time(),
            )
        )

    return state
