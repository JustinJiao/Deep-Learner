# nodes/intent.py
import time
from core.state import AgentState, StepLog
from core.llm_call import run_prompt
from llm.prompts.intent import IntentPrompt
from nodes.log_utils import clip_text, preview_messages


def intent_node(state: AgentState) -> AgentState:
    out = run_prompt(IntentPrompt, state)  # {"intent": {...}}

    intent = out.get("intent", {})
    state["intent"] = intent

    state.setdefault("steps_log", []).append(
        StepLog(
            node="intent",
            info={
                "state": {
                    "query_preview": clip_text(state.get("query", ""), 180),
                },
                "llm_input": {
                    "short_term_memory_preview": clip_text(state.get("short_term_memory", ""), 180),
                    "recent_messages_preview": preview_messages(state.get("recent_messages", [])),
                },
                "llm_output": {
                    "intent": intent,
                },
            },
            timestamp=time.time(),
        )
    )
    return state
