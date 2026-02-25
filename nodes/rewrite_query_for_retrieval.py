import time

from core.llm_call import run_prompt
from core.state import AgentState, StepLog
from llm.prompts.rewrite_retrieval_query import RewriteRetrievalQueryPrompt
from llm.prompts.base import PromptContractError
from nodes.log_utils import clip_text, preview_messages


def _base_query(state: AgentState) -> str:
    return str(state.get("resolved_query") or state.get("query") or "").strip()


def rewrite_query_for_retrieval_node(state: AgentState) -> AgentState:
    base_query = _base_query(state)
    llm_error = ""
    used_fallback = False
    out: dict = {}

    try:
        out = run_prompt(RewriteRetrievalQueryPrompt, state)
    except PromptContractError as e:
        # query 改写失败时不阻断流程，直接使用 base_query 继续检索。
        llm_error = f"{type(e).__name__}: {e}"
        used_fallback = True

    retrieval_query = str(out.get("retrieval_query", "")).strip()

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
                    "llm_error_preview": clip_text(llm_error, 220),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
