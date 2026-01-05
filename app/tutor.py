from .state import AgentState
from services.llm_service import chat_completion

def tutor_node(state: AgentState):
    print("--- [Tutor] 正在通过 LLM 生成启发式教学内容 ---")
    
    context = "\n".join(state['context'])
    user_query = state['messages'][0] # 最初的问题
    
    prompt = f"""
    基于以下参考资料：
    {context}
    
    请回答用户的问题：{user_query}
    
    要求：
    1. 使用类比的方法解释核心概念。
    2. 语言要通俗易懂，适合初学者。
    3. 最后给出一个简单的多选题来检测用户的理解。
    """
    
    # 这里的 System Prompt 可以定义 Tutor 的性格
    system_msg = "你是一位 JHU 的资深教授，擅长用苏格拉底教学法引导学生思考。"
    
    response = chat_completion(prompt, system_msg)
    
    return {"messages": [f"Tutor: {response}"]}