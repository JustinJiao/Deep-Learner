# nodes/compose.py

from core.state import AgentState, StepLog
from core.llm_call import run_prompt
from llm.prompts.compose import ComposePrompt
import time
from nodes.log_utils import clip_text, preview_citations, preview_docs, preview_messages


def compose_node(state: AgentState) -> AgentState:
    state.setdefault("repair_hint", "")

    context_pool = state.get("context_pool", [])

    out = run_prompt(ComposePrompt, state)

    state["response"] = out.get("response", "")
    state["citations"] = out.get("citations", [])

    state.setdefault("steps_log", []).append(
        StepLog(
            node="compose",
            info={
                "state": {
                    "query_preview": clip_text(state.get("query", ""), 180),
                    "is_direct_path": state.get("is_direct_path"),
                    "context_pool_count": len(context_pool),
                },
                "llm_input": {
                    "short_term_memory_preview": clip_text(state.get("short_term_memory", ""), 160),
                    "recent_messages_preview": preview_messages(state.get("recent_messages", [])),
                    "long_term_memory_preview": clip_text(state.get("long_term_memory", ""), 160),
                    "repair_hint_preview": clip_text(state.get("repair_hint", ""), 160),
                    "context_pool_preview": preview_docs(context_pool),
                },
                "llm_output": {
                    "response_preview": clip_text(state["response"], 220),
                    "citations_preview": preview_citations(state["citations"]),
                },
            },
            timestamp=time.time(),
        )
    )

    return state
