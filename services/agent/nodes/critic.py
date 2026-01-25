# services/agent/nodes/critic.py
from services.agent.state import AgentState, AgentStep
from config.settings import ResourceFactory
from llm.prompts import CRITIC_SYSTEM_PROMPT, CRITIC_HUMAN_PROMPT_TEMPLATE, LOG_TEMPLATES

def critic_node(state: AgentState):
    """
    审计节点：对比检索内容与生成内容，检测幻觉与逻辑错误
    """
    print("--- [Node] 执行逻辑审计节点 ---")
    
    llm_service = ResourceFactory.get_llm_service()
    
    # 1. 准备审计素材
    docs = state.get("retrieved_contents", [])
    context_str = "\n\n".join([f"资料{i+1}: {d['content']}" for i, d in enumerate(docs)])
    tutor_response = state.get("response", "")
    
    # 2. 构造审计 Prompt
    human_prompt = CRITIC_HUMAN_PROMPT_TEMPLATE.format(
        context=context_str,
        response=tutor_response
    )
    
    # 3. 调用 LLM 进行审计
    critique_result = llm_service.chat_completion(
        prompt=human_prompt,
        system_prompt=CRITIC_SYSTEM_PROMPT
    )
    
    # 4. 解析审计结果
    is_fail = "[FAIL]" in critique_result.upper()
    
    # 根据结果选择日志模版
    log_key = "critic_fail" if is_fail else "critic_pass"
    
    # 返回更新后的状态
    return {
        "critique": critique_result,
        "is_hallucination": is_fail,  # 🌟 修正：确保这里与 router.py 的判断逻辑一致
        "steps_log": [AgentStep(node="critic", thought=LOG_TEMPLATES[log_key])]
    }