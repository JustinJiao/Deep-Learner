import time

from core.llm_call import run_prompt
from core.state import AgentState, StepLog
from llm.prompts.compose_memory_draft import ComposeMemoryDraftPrompt
from nodes.log_utils import clip_text, preview_messages


def _clamp_confidence(value: object) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))


def _normalize_used_chunks(value: object) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, n)


def _has_memory_signal(state: AgentState) -> bool:
    if int(state.get("ltm_hits_count", 0) or 0) > 0:
        return True
    short_term_memory = str(state.get("short_term_memory", "") or "").strip()
    if short_term_memory and short_term_memory not in {"无", "无相关短期记忆", "None"}:
        return True
    recent_messages = state.get("recent_messages", []) or []
    return len(recent_messages) > 0


def compose_memory_draft_node(state: AgentState) -> AgentState:
    shortcut_no_memory = not _has_memory_signal(state)
    effective_query = str(state.get("resolved_query") or state.get("query", "")).strip()
    prompt_state: AgentState = dict(state)
    prompt_state["query"] = effective_query

    if shortcut_no_memory:
        state["draft_answer"] = "记忆证据不足，需检索外部文档。"
        state["draft_confidence"] = 0.0
        state["used_memory_chunks"] = 0
    else:
        out = run_prompt(ComposeMemoryDraftPrompt, prompt_state)
        state["draft_answer"] = str(out.get("draft_answer", "")).strip()
        state["draft_confidence"] = _clamp_confidence(out.get("confidence", 0.0))
        state["used_memory_chunks"] = _normalize_used_chunks(out.get("used_memory_chunks", 0))

    state.setdefault("steps_log", []).append(
        StepLog(
            node="compose_memory_draft",
            info={
                "state": {
                    "query_preview": clip_text(state.get("query", ""), 180),
                    "effective_query_preview": clip_text(effective_query, 180),
                },
                "llm_input": {
                    "short_term_memory_preview": clip_text(state.get("short_term_memory", ""), 160),
                    "recent_messages_preview": preview_messages(state.get("recent_messages", [])),
                    "long_term_memory_preview": clip_text(state.get("long_term_memory", ""), 160),
                },
                "llm_output": {
                    "draft_answer_preview": clip_text(state.get("draft_answer", ""), 220),
                    "draft_confidence": state.get("draft_confidence", 0.0),
                    "used_memory_chunks": state.get("used_memory_chunks", 0),
                    "shortcut_no_memory": shortcut_no_memory,
                },
            },
            timestamp=time.time(),
        )
    )

    return state
