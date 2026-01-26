import chainlit as cl
import uuid
import time
from services.agent.graph import create_deep_learner_graph
from services.agent.state import AgentState

# ==========================================
# 1. 会话初始化
# ==========================================

@cl.on_chat_start
async def start():
    """
    当用户进入 Deep-Learner 界面时，初始化 Graph 和 Session。
    """
    # 编译并加载你设计的 LangGraph 状态机
    app = create_deep_learner_graph()
    
    # 将持久化对象存入 Session
    cl.user_session.set("graph", app)
    cl.user_session.set("thread_id", str(uuid.uuid4()))
    
    await cl.Message(
        content="👋 **Deep-Learner** 数字导师已上线。我已准备好基于本地知识库为您解答技术难题。"
    ).send()

# ==========================================
# 2. 核心交互逻辑
# ==========================================

@cl.on_message
async def main(message: cl.Message):
    """
    接收用户 Query，驱动节点执行，并渲染最终的溯源回答。
    """
    graph = cl.user_session.get("graph")
    thread_id = cl.user_session.get("thread_id")
    
    # 构造初始状态
    initial_state = {
        "query": message.content,
        "steps_log": [],
        "retrieved_contents": [],
        "source_mapping": {},  # 🌟 接收来自 retriever 节点的映射数据
        "response": "",
        "metadata": {}
    }

    # 1. 执行 Agent 逻辑 (Invoke 模式)
    # 如果面试时想演示“思考流”，可后续开启 astream_events
    final_state = await cl.make_async(graph.invoke)(
        initial_state, 
        config={"configurable": {"thread_id": thread_id}}
    )

    # 2. 渲染节点思考链路 (Steps)
    # 这能向面试官直观展示 Planner -> Retriever -> Tutor 的决策逻辑
    for step_data in final_state.get("steps_log", []):
        async with cl.Step(name=step_data.node.upper()) as step:
            step.output = step_data.thought

    # 3. 构建溯源元素 (Citations)
    # 关键：利用 source_mapping 把 ID 还原回干净的文件名
    source_elements = []
    mapping = final_state.get("source_mapping", {})
    retrieved_docs = final_state.get("retrieved_contents", [])

    for doc in retrieved_docs:
        doc_id = doc.get("id")
        # 从映射表中取出“脱敏”后的文件名
        clean_file_name = mapping.get(doc_id, "未知文档")
        
        # 元素名称必须匹配 [Source N] 中的 "Source N"
        element_name = f"Source {doc_id}"
        
        # 组装侧边栏显示的内容
        display_content = (
            f"📄 **原始文档**: {clean_file_name}\n\n"
            f"---\n\n"
            f"{doc['content']}"
        )
        
        source_elements.append(
            cl.Text(name=element_name, content=display_content, display="side")
        )

    # 4. 发送最终答案
    response_text = final_state.get("response", "抱歉，导师未能生成有效回答。")
    
    # 如果是 Critic 审计失败后的结果，增加一个警告标志
    if final_state.get("is_hallucination"):
        response_text = "⚠️ **内容可能存在幻觉**\n\n" + response_text

    await cl.Message(
        content=response_text,
        elements=source_elements  # 👈 绑定点击关联
    ).send()