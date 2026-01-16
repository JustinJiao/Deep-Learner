from utils.json_utils import safe_parse_json
from services.llm_service import chat_completion

def tutor_node(state):
    print("--- [Tutor] 正在基于资料生成教学内容 ---")
    context = "\n".join(state.get('context', []))
    user_query = state['messages'][0]
    
    prompt = f"""你是一个资深教授。请根据参考资料回答用户问题。
    【参考资料】：{context if context else "【暂无相关资料】"}
    【用户问题】：{user_query}
    
    【行为准则】：
    1. 如果参考资料与问题无关或为空，status 必须设为 "no_data"。
    2. 只有在资料充足时，status 才设为 "success"。
    3. 严禁使用参考资料以外的知识回答核心技术点。
    
    你必须只输出如下 JSON 格式：
    {{
        "status": "success/no_data",
        "answer": "教学内容或拒绝回答的说明"
    }}"""
    
    response = chat_completion(prompt, "You are a helpful and honest teacher.")
    return {"answer": safe_parse_json(response)}