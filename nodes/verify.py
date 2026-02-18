# nodes/verify.py
from core.state import AgentState, StepLog
from core.llm_call import run_prompt
from llm.prompts.verify import VerifyPrompt


def verify_node(state: AgentState) -> AgentState:
    out = run_prompt(VerifyPrompt, state)

    state["is_hallucination"] = bool(out["is_hallucination"])
    state["critique"] = out  # dict with routing fields

    # PASS 时清掉旧的 repair_hint，避免污染后续轮次
    if not state["is_hallucination"]:
        state.pop("repair_hint", None)

    verdict = "FAIL" if state["is_hallucination"] else "PASS"
    state.setdefault("steps_log", []).append(
        StepLog(node="verify", info=f"{verdict} type={out.get('error_type')} next={out.get('next_step')}")
    )
    return state
