from langgraph.graph import StateGraph, END
from services.agent.state import AgentState
from services.agent.nodes import (
    planner_node, 
    retriever_node, 
    tutor_node, 
    critic_node,
    rewriter_node,
    finalizer_node
)
from services.agent.edges.router import check_retrieval_loop, check_audit_loop
from langgraph.checkpoint.memory import MemorySaver  # 🌟 1. 引入内存保存器
# --- 图构建 ---
def create_deep_learner_graph():
    workflow = StateGraph(AgentState)

    # 注册节点
    workflow.add_node("rewriter", rewriter_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("tutor", tutor_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("finalizer", finalizer_node)
    memory = MemorySaver()  # 🌟 2. 创建内存保存器实例

    # 设置路径
    workflow.set_entry_point("rewriter")
    workflow.add_edge("rewriter", "planner")
    workflow.add_edge("planner", "retriever")
    # 迭代检索循环 (根据任务步骤数循环)
    workflow.add_conditional_edges(
        "retriever",
        check_retrieval_loop,
        {"continue_retrieval": "retriever", "go_tutor": "tutor"}
    )
    
    workflow.add_edge("tutor", "critic")
    
    # 审计反馈循环
    workflow.add_conditional_edges(
        "critic",
        check_audit_loop,
        {"re_plan": "planner", "finish": "finalizer"}
    )
    
    workflow.add_edge("finalizer", END)
    return workflow.compile(checkpointer=memory)  # 🌟 3. 将内存保存器传递给图编译器