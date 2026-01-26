# services/agent/nodes/planner.py
from services.agent.state import AgentState, AgentStep
from config.settings import ResourceFactory
from llm.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_HUMAN_PROMPT_TEMPLATE

def planner_node(state: AgentState):
    """
    规划节点：从 llm/prompts 引用提示词，解耦逻辑与内容
    """
    print("--- [Node] 执行规划节点 ---")
    
    llm_service = ResourceFactory.get_llm_service()
    
    # 动态注入参数
    human_prompt = PLANNER_HUMAN_PROMPT_TEMPLATE.format(query=state['query'])
    
    raw_plan = llm_service.chat_completion(
        prompt=human_prompt, 
        system_prompt=PLANNER_SYSTEM_PROMPT
    )
    
    plan_list = [p.strip() for p in raw_plan.split("\n") if p.strip()]
    
    return {
        "plan": plan_list,
        "current_step_idx": 0,
        "steps_log": [AgentStep(node="planner", thought=f"拆解目标: {plan_list}")]
    }