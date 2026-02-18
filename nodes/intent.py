# nodes/intent.py
import time
from core.state import AgentState, StepLog
from core.llm_call import run_prompt
from llm.prompts.intent import IntentPrompt


def intent_node(state: AgentState) -> AgentState:
    out = run_prompt(IntentPrompt, state)  # {"intent": {...}}

    state["intent"] = out["intent"]

    state.setdefault("steps_log", []).append(
        StepLog(node="intent", info=f"type={out['intent']['type']}", timestamp=time.time())
    )
    return state
