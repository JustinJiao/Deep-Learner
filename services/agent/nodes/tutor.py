# services/agent/nodes/tutor.py
from services.agent.state import AgentState, AgentStep
from config.settings import ResourceFactory
from llm.prompts import TUTOR_SYSTEM_PROMPT, TUTOR_HUMAN_PROMPT_TEMPLATE, LOG_TEMPLATES

def tutor_node(state: AgentState):
    """
    导师节点：将检索到的原始知识块加工成结构化教学内容
    """
    print("--- [Node] 执行导师教学节点 ---")
    
    llm_service = ResourceFactory.get_llm_service()
    
    # 1. 整理参考资料 (Context Formatting)
    # 我们利用之前 Ingestion 阶段存入的 metadata 进行格式化
    docs = state.get("retrieved_contents", [])
    context_list = []
    for i, doc in enumerate(docs):
        meta = doc.get("metadata", {})
        source_info = f"来源: {meta.get('source', '未知')} | 章节: {meta.get('h2', 'Intro')}"
        context_list.append(f"[{i+1}] {source_info}\n内容: {doc['content']}")
    
    context_str = "\n\n".join(context_list)
    
    # 2. 构造 Prompt
    human_prompt = TUTOR_HUMAN_PROMPT_TEMPLATE.format(
        context=context_str,
        query=state['query']
    )
    
    # 3. 调用 LLM 生成
    # 注意：这里可以根据需要调高 temperature (如 0.7)，让导师说话更自然
    response = llm_service.chat_completion(
        prompt=human_prompt,
        system_prompt=TUTOR_SYSTEM_PROMPT
    )
    
    return {
        "response": response,
        "steps_log": [AgentStep(node="tutor", thought=LOG_TEMPLATES["tutor_success"])]
    }