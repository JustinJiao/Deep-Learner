import time
from services.agent.state import AgentState, AgentStep
from config.settings import ResourceFactory
from llm.prompts import TUTOR_SYSTEM_PROMPT, TUTOR_HUMAN_PROMPT_TEMPLATE, LOG_TEMPLATES

def tutor_node(state: AgentState):
    """
    导师节点：只向 LLM 展示 ID 和内容，完全隔离物理路径。
    """
    print("--- [Node] 执行导师教学节点 ---")
    llm_service = ResourceFactory.get_llm_service()
    
    # 1. 整理参考资料 (只给模型看 ID)
    docs = state.get("retrieved_contents", [])
    context_list = []
    
    for doc in docs:
        doc_id = doc.get("id")
        # 🌟 核心安全操作：只传递逻辑 ID
        context_block = f"[[资料编号: {doc_id}]]\n内容: {doc['content']}"
        context_list.append(context_block)
    
    context_str = "\n\n".join(context_list)
    # print(f'--- 整理后传递给 LLM 的参考资料 ---\n{context_str}\n--- 结束 ---')
    # 2. 构造 Prompt
    human_prompt = TUTOR_HUMAN_PROMPT_TEMPLATE.format(
        context=context_str,
        query=state['query']
    )
    
    # 3. 调用 LLM 生成回答
    response = llm_service.chat_completion(
        prompt=human_prompt,
        system_prompt=TUTOR_SYSTEM_PROMPT
    )
    print(f"--- 导师节点生成的回答 ---\n{response}\n--- 结束 ---")
    return {
        "response": response,
        "steps_log": [AgentStep(node="tutor", thought=LOG_TEMPLATES["tutor_success"], timestamp=time.time())]
    }