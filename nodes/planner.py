# nodes/planner.py
from core.state import AgentState, StepLog
from core.plan import ExecutionPlan
from config.settings import AppConfig


def planner_node(state: AgentState) -> AgentState:
    
    intent_type = state["intent"]["type"]

    if intent_type == "chat":
        steps = ["compose"]
        is_direct = True
    else:
        steps = ["recall_ltm", "query_rewrite", "retrieve", "compose", "verify"]
        is_direct = False

    state["plan"] = ExecutionPlan(
        steps=steps,
        step_idx=0,
        max_loops=int(getattr(AppConfig, "MAX_REPAIR_LOOPS", 3)),
    )

    state["is_direct_path"] = is_direct
    state["loop_count"] = 0
    state.setdefault("repair_hint", "")
    # 保证 compose 所需字段存在（防止 PromptContractError）
    state.setdefault("rewritten_query", "")
    state.setdefault("context_pool", [])
    state.setdefault("long_term_memory", "")
    state.setdefault("repair_hint", "")

    state.setdefault("steps_log", []).append(
        StepLog(node="planner", info=f"plan={steps}, direct={is_direct}")
    )
    return state
