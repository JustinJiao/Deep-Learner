# nodes/planner.py
import time

from core.state import AgentState, StepLog
from core.plan import ExecutionPlan
from config.settings import AppConfig
from nodes.log_utils import clip_text


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
        max_loops=int(AppConfig.MAX_REPAIR_LOOPS),
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
        StepLog(
            node="planner",
            info={
                "state": {
                    "query_preview": clip_text(state.get("query", ""), 180),
                    "intent": state.get("intent", {}),
                    "is_direct_path": is_direct,
                    "loop_count": state.get("loop_count"),
                    "plan": {
                        "steps": steps,
                        "step_idx": state["plan"].step_idx,
                        "current_step": state["plan"].current_step() if steps else None,
                        "max_loops": state["plan"].max_loops,
                    },
                },
            },
            timestamp=time.time(),
        )
    )
    return state
