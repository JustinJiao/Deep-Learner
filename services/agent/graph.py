from langgraph.graph import StateGraph, END
from services.agent.state import AgentState
from services.agent.nodes import (
    planner_node, 
    retriever_node, 
    tutor_node, 
    critic_node
)
from services.agent.edges.router import grade_response_route

def create_deep_learner_graph():
    """
    编排 Deep-Learner 的 Agentic RAG 流程图。
    实现了：任务拆解 -> 混合检索 -> 教学生成 -> 逻辑反思 的闭环。
    """
    # 1. 声明一个基于 AgentState 的状态图
    workflow = StateGraph(AgentState)

    # 2. 注册所有节点
    workflow.add_node("planner", planner_node)      # 目标拆解
    workflow.add_node("retriever", retriever_node)  # 知识召回 (对接 retrieval 模块)
    workflow.add_node("tutor", tutor_node)          # 启发式教学 (对接 LLM 模块)
    workflow.add_node("critic", critic_node)        # 逻辑审计

    # 3. 设定默认执行路径
    workflow.set_entry_point("planner")             # 入口：先做规划
    workflow.add_edge("planner", "retriever")       # 规划完去搜资料
    workflow.add_edge("retriever", "tutor")         # 搜完资料开始教课
    workflow.add_edge("tutor", "critic")           # 教完后接受审计

    # 4. 设定条件路由 (实现循环/退出逻辑)
    workflow.add_conditional_edges(
        "critic",                                   # 从审计节点开始判断
        grade_response_route,                       # 调用决策函数
        {
            "re_plan": "planner",                   # 逻辑有问题，回炉重造
            "finish": END                           # 逻辑满分，直接输出
        }
    )

    # 5. 编译图 (后续可在此处增加 checkpointer 实现断点续传)
    return workflow.compile()