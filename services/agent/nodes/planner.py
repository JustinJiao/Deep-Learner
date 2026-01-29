from services.agent.state import AgentState, AgentStep
from config.settings import ResourceFactory
from llm.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_HUMAN_PROMPT_TEMPLATE, PLANNER_RETRY_HUMAN_PROMPT    
import time
def planner_node(state: AgentState):
    print("--- [Node] 任务拆解与目标规划 ---")
    llm = ResourceFactory.get_llm_service()
    loop_count = state.get("loop_count", 0)
    
    # 如果是重试，注入审计反馈
    if loop_count > 0 and state.get("critique"):
        prompt = PLANNER_RETRY_HUMAN_PROMPT.format(query=state['rewritten_query'], critique=state['critique'])
    else:
        prompt = PLANNER_HUMAN_PROMPT_TEMPLATE.format(query=state['rewritten_query'])
    
    raw_plan = llm.chat_completion(prompt=prompt, system_prompt=PLANNER_SYSTEM_PROMPT)
    plan_list = [p.strip().lstrip("- 123456789. ") for p in raw_plan.split("\n") if p.strip()]
    
    return {
        "plan": plan_list,
        "current_step_idx": 0,
        "context_pool": [], # 🌟 重新规划时清空旧知识池
        "source_mapping": {},
        "steps_log": [AgentStep(node="planner", thought=f"拆解子任务: {plan_list}", timestamp=time.time())]
    }