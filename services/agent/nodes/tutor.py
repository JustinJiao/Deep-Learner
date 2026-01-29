import time
from services.agent.state import AgentState, AgentStep
from config.settings import ResourceFactory
from llm.prompts import TUTOR_SYSTEM_PROMPT, TUTOR_HUMAN_PROMPT_TEMPLATE, LOG_TEMPLATES, TUTOR_FALLBACK_SYSTEM_PROMPT, TUTOR_DISCLAIMER_TEXT

def tutor_node(state: AgentState):
    print("--- [Node] 知识整合与教学回答生成 ---")
    llm = ResourceFactory.get_llm_service()
    # 对齐：使用汇总后的 context_pool
    docs = state.get("context_pool", [])
    context_list = []
    for d in docs:
        context_list.append(f"--- 文档名称: {d['id']} ---\n内容: {d['content']}")
    context_str = "\n\n".join(context_list)
    
    is_fallback = (state.get("loop_count", 0) >= 2)
    sys_prompt = TUTOR_SYSTEM_PROMPT + (TUTOR_FALLBACK_SYSTEM_PROMPT if is_fallback else "")

    prompt = TUTOR_HUMAN_PROMPT_TEMPLATE.format(
        context=context_str if context_str else "未获取到直接资料。",
        query=state['rewritten_query'] # 使用改写后的标准问题
    )
    
    response = llm.chat_completion(prompt=prompt, system_prompt=sys_prompt)
    if is_fallback: response = TUTOR_DISCLAIMER_TEXT + response

    return {
        "response": response,
        "steps_log": [AgentStep(node="tutor", thought="生成综合教学回答", timestamp=time.time())]
    }