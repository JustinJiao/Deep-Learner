# nodes/repair.py
from core.state import AgentState, StepLog


def repair_node(state: AgentState) -> AgentState:
    """
    根据 verify 给出的错误类型与 next_step 做“路由式修复”。

    - 检索没问题但生成有问题 -> 回到 compose（复用 context_pool）
    - 文档不足 -> 回到 retrieve（清空旧 context_pool）
    - 检索方向错/歧义 -> 回到 query_rewrite（清空旧 context_pool）
    """
    critique = state.get("critique") or {}
    next_step = critique.get("next_step", "compose")
    error_type = critique.get("error_type", "UNKNOWN")
    hint = critique.get("critique", "")

    # 写给 compose 的修复提示
    state["repair_hint"] = f"[{error_type}] {hint}".strip()

    # 路由 + 防御
    if next_step == "compose":
        state["plan"].jump_to("compose")
    elif next_step == "retrieve":
        # 重新检索前清空旧文档池，避免复用过期结果
        state.pop("context_pool", None)
        state["plan"].jump_to("retrieve")
    elif next_step == "query_rewrite":
        state.pop("context_pool", None)
        state["plan"].jump_to("query_rewrite")
    else:
        state["plan"].jump_to("compose")
        next_step = "compose"

    state.setdefault("steps_log", []).append(
        StepLog(node="repair", info=f"route={next_step} type={error_type}")
    )
    return state
