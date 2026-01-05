from .state import AgentState
def critic_node(state: AgentState):
    print("--- [Critic] 逻辑校验 ---")
    # V1: 模拟一个随机校验逻辑
    is_valid = len(state['context']) > 0
    feedback = "通过校验" if is_valid else "检索内容不足，需要重新规划"
    return {"is_valid": is_valid, "messages": [f"Critic: {feedback}"]}