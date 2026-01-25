from services.agent.state import AgentState

def grade_response_route(state: AgentState):
    """
    路由函数：判断当前生成内容的质量，决定是结束还是重试。
    """
    print(f"--- [Deep-Learner Router] 正在评估节点: {state.get('steps_log')[-1].node} ---")
    
    # 获取审计结果
    is_bad = state.get("is_hallucination", False)
    critique = state.get("critique", "PASS")

    # 如果存在幻觉或审计未通过，则返回到 Planner 重新规划检索策略
    if is_bad or critique == "FAIL":
        print("--- [Router] 检测到内容存在逻辑缺陷，指令：重修计划并重新检索 ---")
        return "re_plan"
    
    print("--- [Router] 质量校验通过，准备输出最终响应 ---")
    return "finish"