from utils.json_utils import safe_parse_json
from services.llm_service import chat_completion

def critic_node(state):
    print("--- [Critic] 正在进行一致性审计 ---")
    user_query = state['messages'][0]
    context = str(state.get('context', []))
    tutor_res = state['answer']
    
    # 强化版的审计提示词
    prompt = f"""你是一名极其严苛的 RAG 教学质量审计员。你的任务是防止系统在检索到错误资料时编造答案。
    
    【核心审计三元组】：
    1. 用户问题 (Query): {user_query}
    2. 检索到的原始资料 (Context): {context}
    3. 助手生成的回答 (Answer): {tutor_res['answer']}
    
    【强制审计准则】：
    1. 相关性校验 (Relevance Check)：检索到的【知识库资料】是否真的包含回答【用户问题】所需的关键信息？
       - 如果资料在聊“Milvus/向量数据库”，但问题是“Spark”，必须判定为 False。
    2. 忠实度校验 (Faithfulness Check)：【助手回答】中的每一个核心技术点，是否都能在【知识库资料】中找到出处？
       - 如果回答中出现了资料里根本没提到的词（如：RowBuffer, RDD），即使这些知识是正确的，也必须判定为 False (因为这属于脱离资料的幻觉)。
    3. 诚实度校验 (Honesty Check)：如果资料与问题不匹配，助手应该承认不知道。如果助手强行回答，判定为 False。
    
    你必须严格按 JSON 格式输出：
    {{
        "is_valid": false/true,
        "reason": "请详细描述：1.资料是否跑题；2.回答是否包含了资料之外的幻觉内容。"
    }}"""
    
    response = chat_completion(prompt, "You are a strict technical auditor.")
    audit_data = safe_parse_json(response)
    
    # 逻辑控制：如果不通过，可以在这里记录原因，供 Planner 参考
    return {
        "is_valid": audit_data['is_valid'],
        "revision_notes": audit_data['reason'],
        "retry_count": state.get('retry_count', 0) + 1
    }