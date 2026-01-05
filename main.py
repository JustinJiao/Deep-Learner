from langgraph.graph import StateGraph, END
from app.state import AgentState
from app.planner import planner_node
from app.retriever import retriever_node
from app.tutor import tutor_node
from app.critic import critic_node

# 1. 初始化图
workflow = StateGraph(AgentState)

# 2. 添加节点
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("tutor", tutor_node)
workflow.add_node("critic", critic_node)

# 3. 设置入口
workflow.set_entry_point("planner")

# 4. 构建边（连线）
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "tutor")
workflow.add_edge("tutor", "critic")

# 5. 条件循环 (这就是 Agentic 的灵魂)
def decide_to_end(state):
    if state["is_valid"] or state.get("retry_count", 0) > 2:
        return "end"
    else:
        return "retry"

workflow.add_conditional_edges(
    "critic",
    decide_to_end,
    {
        "end": END,
        "retry": "planner" # 校验失败，回到 planner 重新规划
    }
)

# 6. 编译并运行
app = workflow.compile()

if __name__ == "__main__":
    inputs = {"messages": ["我想学习 Spark 内存管理"], "retry_count": 0}
    for output in app.stream(inputs):
        print(output)