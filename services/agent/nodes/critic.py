# services/agent/nodes/critic.py
from services.agent.state import AgentState, AgentStep
from config.settings import ResourceFactory
from llm.prompts import CRITIC_SYSTEM_PROMPT, CRITIC_HUMAN_PROMPT_TEMPLATE, LOG_TEMPLATES
import time
def critic_node(state: AgentState):
    """5. 逻辑审计：多维度校验"""
    print("--- [Node] 逻辑审计与回答校验 ---")
    llm = ResourceFactory.get_llm_service()
    context_str = str(state["context_pool"])
    
    prompt = CRITIC_HUMAN_PROMPT_TEMPLATE.format(context=context_str, response=state['response'])
    result = llm.chat_completion(prompt=prompt, system_prompt=CRITIC_SYSTEM_PROMPT)
    
    is_fail = "[FAIL]" in result.upper()
    current_loop = state.get("loop_count", 0)
    return {
        "critique": result,
        "is_hallucination": is_fail,
        "loop_count": current_loop + 1 if is_fail else current_loop,
        "steps_log": [AgentStep(node="critic", thought=f"审计结果: {'未通过' if is_fail else '通过'}", timestamp=time.time())]
    }