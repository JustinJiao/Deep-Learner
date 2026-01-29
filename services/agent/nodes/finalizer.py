from services.agent.state import AgentState, AgentStep
import time
def finalizer_node(state: AgentState):
    """
    归档节点：记录对话，清空工作区，实现记忆滑动窗口。
    """
    print("--- [Node] 最终归档与状态重置 ---")
    
    # 1. 记忆滑动窗口处理
    current_msgs = state.get("messages", [])
    if len(current_msgs) > 10:
        current_msgs = current_msgs[-10:]
    
    # 2. 返回更新：显式将临时状态置空
    return {
        "messages": [
            ("user", state['query']), 
            ("assistant", state['response'])
        ],
        "plan": [],
        "current_step_idx": 0,
        "context_pool": [],
        "loop_count": 0,
        "critique": None,
        "is_hallucination": False,
        "steps_log": [AgentStep(node="finalizer", thought="状态机已重置，等待新任务", timestamp=time.time())]
    }