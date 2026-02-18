# nodes/persist_ltm.py
from core.state import AgentState, StepLog
from memory.ltm import LTM
from core.llm_call import run_prompt
from llm.prompts.ltm_fact_extract import LTMFactExtractPrompt


def persist_ltm_node(state: AgentState) -> AgentState:
    # 没有 response 就不抽取，避免污染
    if not state.get("response"):
        return state

    out = run_prompt(LTMFactExtractPrompt, state)
    candidates = out.get("fact_candidates", []) or []

    ltm = LTM()
    stored = ltm.upsert(candidates)

    state.setdefault("steps_log", []).append(
        StepLog(node="persist_ltm", info=f"stored={stored}")
    )
    return state
