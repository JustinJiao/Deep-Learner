import time

from core.llm_call import run_prompt
from core.state import AgentState, StepLog
from llm.prompts.resolve_query_reference import ResolveQueryReferencePrompt
from llm.prompts.base import PromptContractError
from nodes.log_utils import clip_text, preview_messages


def resolve_query_reference_node(state: AgentState) -> AgentState:
    original_query = str(state.get("query", "")).strip()
    llm_error = ""
    used_fallback = False
    out: dict = {}

    try:
        out = run_prompt(ResolveQueryReferencePrompt, state)
    except PromptContractError as e:
        # JSON 解析失败等格式问题时，不中断主流程，直接回退到原 query。
        llm_error = f"{type(e).__name__}: {e}"
        used_fallback = True

    resolved_query = str(out.get("resolved_query", "")).strip()

    if not resolved_query:
        resolved_query = original_query
        used_fallback = True

    state["resolved_query"] = resolved_query

    state.setdefault("steps_log", []).append(
        StepLog(
            node="resolve_query_reference",
            info={
                "state": {
                    "query_preview": clip_text(original_query, 180),
                },
                "llm_input": {
                    "short_term_memory_preview": clip_text(state.get("short_term_memory", ""), 160),
                    "recent_messages_preview": preview_messages(state.get("recent_messages", [])),
                },
                "llm_output": {
                    "resolved_query_preview": clip_text(resolved_query, 180),
                    "used_fallback": used_fallback,
                    "llm_error_preview": clip_text(llm_error, 220),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
