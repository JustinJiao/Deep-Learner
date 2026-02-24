import time

from core.llm_call import run_prompt
from core.state import AgentState, StepLog
from llm.prompts.rewrite_retrieval_query import RewriteRetrievalQueryPrompt
from nodes.log_utils import clip_text, preview_messages


def _base_query(state: AgentState) -> str:
    return str(state.get("resolved_query") or state.get("query") or "").strip()


def rewrite_query_for_retrieval_node(state: AgentState) -> AgentState:
    base_query = _base_query(state)
    out = run_prompt(RewriteRetrievalQueryPrompt, state)
    retrieval_query = str(out.get("retrieval_query", "")).strip()

    used_fallback = False
    if not retrieval_query:
        retrieval_query = base_query
        used_fallback = True

    state["retrieval_query"] = retrieval_query

    state.setdefault("steps_log", []).append(
        StepLog(
            node="rewrite_query_for_retrieval",
            info={
                "state": {
                    "query_preview": clip_text(str(state.get("query", "")), 180),
                    "resolved_query_preview": clip_text(str(state.get("resolved_query", "")), 180),
                },
                "llm_input": {
                    "long_term_memory_preview": clip_text(state.get("long_term_memory", ""), 160),
                    "recent_messages_preview": preview_messages(state.get("recent_messages", [])),
                },
                "llm_output": {
                    "base_query_preview": clip_text(base_query, 180),
                    "retrieval_query_preview": clip_text(retrieval_query, 180),
                    "used_fallback": used_fallback,
                },
            },
            timestamp=time.time(),
        )
    )
    return state
