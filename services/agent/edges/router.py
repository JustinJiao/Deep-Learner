from services.agent.state import AgentState
from services.agent.nodes import *

# --- 路由函数 ---
def check_retrieval_loop(state: AgentState):
    """判断子任务是否搜完"""
    if state["current_step_idx"] < len(state["plan"]):
        return "continue_retrieval"
    return "go_tutor"

def check_audit_loop(state: AgentState):
    """判断审计是否通过或达到重试上限"""
    if state["is_hallucination"] and state["loop_count"] < 3:
        return "re_plan"
    return "finish"