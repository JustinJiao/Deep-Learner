# nodes/compose.py
from core.state import AgentState, StepLog
from core.llm_call import run_prompt
from llm.prompts.compose import ComposePrompt


def compose_node(state: AgentState) -> AgentState:
    state.setdefault("repair_hint", "")
    out = run_prompt(ComposePrompt, state)
    state["response"] = out["response"]

    state.setdefault("steps_log", []).append(
        StepLog(node="compose", info="generated")
    )
    return state
