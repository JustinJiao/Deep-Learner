import time

from config.settings import AppConfig
from core.llm_call import run_prompt
from core.state import AgentState, StepLog
from llm.prompts.base import PromptContractError
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


def _apply_memory_fallback(state: AgentState) -> None:
    state["draft_answer"] = "记忆证据不足，需检索外部文档。"
    state["draft_confidence"] = 0.0
    state["used_memory_chunks"] = 0


def compose_memory_draft_node(state: AgentState) -> AgentState:
    shortcut_no_memory = not _has_memory_signal(state)
    effective_query = str(state.get("resolved_query") or state.get("query", "")).strip()
    prompt_state: AgentState = dict(state)
    prompt_state["query"] = effective_query
    llm_attempts = 0
    llm_error = ""
    used_fallback = False

    if shortcut_no_memory:
        _apply_memory_fallback(state)
    else:
        max_retries = max(0, int(AppConfig.RUNTIME_MEMORY_DRAFT_MAX_RETRIES))
        total_attempts = 1 + max_retries
        out: dict | None = None

        for attempt in range(1, total_attempts + 1):
            llm_attempts = attempt
            try:
                out = run_prompt(ComposeMemoryDraftPrompt, prompt_state)
                break
            except PromptContractError as e:
                llm_error = f"{type(e).__name__}: {e}"
                if attempt >= total_attempts:
                    used_fallback = True
            except Exception as e:  # pragma: no cover
                # memory draft 失败不应打断主流程，直接降级到检索路径
                llm_error = f"{type(e).__name__}: {e}"
                used_fallback = True
                break

        if out is None:
            _apply_memory_fallback(state)
            used_fallback = True
        else:
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
                    "llm_attempts": llm_attempts,
                    "used_fallback": used_fallback,
                    "llm_error_preview": clip_text(llm_error, 220),
                },
            },
            timestamp=time.time(),
        )
    )

    return state
