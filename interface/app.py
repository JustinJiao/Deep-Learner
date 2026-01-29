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
    graph = cl.user_session.get("graph")
    thread_id = cl.user_session.get("thread_id")
    
    # 1. 🌟 状态协议对齐
    initial_state = {
        "query": message.content,
        "loop_count": 0,           # 必须初始化，防止 Critic 节点报错
        "current_step_idx": 0,     # 确保从第一步开始检索
        "context_pool": [],        # 替换旧的 retrieved_contents
        "source_mapping": {},      
        "steps_log": [],
        "messages": []             # 传入空列表，MemorySaver 会自动合并历史记忆
    }

    # 2. 执行推理
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    final_state = await cl.make_async(graph.invoke)(initial_state, config=config)

    # 3. 渲染思考链路
    for step_data in final_state.get("steps_log", []):
        # 兼容处理：确保能读取 Pydantic 对象属性
        node_name = step_data.node if hasattr(step_data, 'node') else "AGENT"
        thought_text = step_data.thought if hasattr(step_data, 'thought') else str(step_data)
        
        async with cl.Step(name=node_name.upper()) as step:
            step.output = thought_text

    # 4. 🌟 溯源元素构建 (基于文件名)
    source_elements = []
    # 使用对齐后的 context_pool
    retrieved_docs = final_state.get("context_pool", [])

    for doc in retrieved_docs:
        file_name = doc.get("id")  # 此时 ID 就是文件名
        
        # 🌟 关键：element_name 必须与导师回答里的引用文字完全一致
        # 如果导师回答 [test_pdf.pdf]，这里必须叫 "test_pdf.pdf"
        element_name = file_name
        
        display_content = (
            f"📄 **源文件**: {file_name}\n\n"
            f"---\n\n"
            f"{doc['content']}"
        )
        
        source_elements.append(
            cl.Text(name=element_name, content=display_content, display="side")
        )

    # 5. 发送最终答案
    response_text = final_state.get("response", "抱歉，导师未能生成有效回答。")
    
    if final_state.get("is_hallucination"):
        response_text = "⚠️ **[逻辑审计] 内容可能存在引用偏差**\n\n" + response_text

    await cl.Message(
        content=response_text,
        elements=source_elements
    ).send()