import time
from services.agent.state import AgentState, AgentStep
from config.settings import ResourceFactory
from llm.prompts import REWRITER_SYSTEM_PROMPT, REWRITER_HUMAN_PROMPT_TEMPLATE

def rewriter_node(state: AgentState):
    print("--- [Node] 意图对齐与问题改写 ---")
    llm = ResourceFactory.get_llm_service()
    # 提取滑动窗口历史 (最近 10 条)
    history = "\n".join([f"{m[0]}: {m[1]}" for m in state.get("messages", [])[-10:]])
    
    prompt = REWRITER_HUMAN_PROMPT_TEMPLATE.format(query=state['query'], history=history or "无")
    rewritten = llm.chat_completion(prompt=prompt, system_prompt=REWRITER_SYSTEM_PROMPT).strip()
    
    return {
        "rewritten_query": rewritten,
        "steps_log": [AgentStep(node="rewriter", thought=f"意图对齐：{rewritten}", timestamp=time.time())]
    }