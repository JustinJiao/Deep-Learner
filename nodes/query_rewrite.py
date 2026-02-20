# nodes/query_rewrite.py
import time
from core.state import AgentState, StepLog
from core.llm_call import run_prompt
from llm.prompts.rewrite import RewritePrompt
from nodes.log_utils import clip_text


def query_rewrite_node(state: AgentState) -> AgentState:
    original_query = state.get("query", "")
    out = run_prompt(RewritePrompt, state)  # {"rewritten_query": "..."}
    rewritten_query = out.get("rewritten_query", "")
    state["rewritten_query"] = rewritten_query

    state.setdefault("steps_log", []).append(
        StepLog(
            node="query_rewrite",
            info={
                "state": {
                    "query_preview": clip_text(original_query, 180),
                },
                "llm_input": {
                    "long_term_memory_preview": clip_text(state.get("long_term_memory", ""), 180),
                },
                "llm_output": {
                    "rewritten_query_preview": clip_text(rewritten_query, 180),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
