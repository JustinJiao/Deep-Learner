from .state import AgentState
def planner_node(state: AgentState):
    user_query = state['messages'][-1]
    # V1: 简单模拟拆解逻辑
    print(f"--- [Planner] 正在拆解任务: {user_query} ---")
    return {
        "plan": f"搜索关于 {user_query} 的核心概念和常见误区", 
        "messages": ["Planner: 我已制定学习路径"]
    }