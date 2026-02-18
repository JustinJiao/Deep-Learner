# nodes/query_rewrite.py
import time
from core.state import AgentState, StepLog
from core.llm_call import run_prompt
from llm.prompts.rewrite import RewritePrompt


def query_rewrite_node(state: AgentState) -> AgentState:
    out = run_prompt(RewritePrompt, state)  # {"rewritten_query": "..."}
    state["rewritten_query"] = out["rewritten_query"]

    state.setdefault("steps_log", []).append(
        StepLog(node="query_rewrite", info="ok", timestamp=time.time())
    )
    return state
