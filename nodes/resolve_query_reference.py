import time

from core.llm_call import run_prompt
from core.state import AgentState, StepLog
from llm.prompts.resolve_query_reference import ResolveQueryReferencePrompt
from nodes.log_utils import clip_text, preview_messages


def resolve_query_reference_node(state: AgentState) -> AgentState:
    original_query = str(state.get("query", "")).strip()
    out = run_prompt(ResolveQueryReferencePrompt, state)
    resolved_query = str(out.get("resolved_query", "")).strip()

    used_fallback = False
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
                },
            },
            timestamp=time.time(),
        )
    )
    return state
