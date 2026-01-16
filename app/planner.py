from .state import AgentState
from services.llm_service import chat_completion
from utils.json_utils import safe_parse_json

# app/planner.py 修改点

def planner_node(state):
    print("\n--- [Planner] 正在生成检索计划 ---")
    user_query = state['messages'][0]
    
    # 获取重试次数和历史建议
    retry_count = state.get('retry_count', 0)
    feedback = state.get('revision_notes', '这是初次尝试。')
    
    # 如果是重试，强化“变通”的要求
    retry_instruction = ""
    if retry_count > 0:
        retry_instruction = f"""
        【重要：这是第 {retry_count} 次尝试】
        之前的检索失败了！反馈如下：{feedback}
        
        你的任务是“变通”：
        1. 不要使用已经失败过的关键词。
        2. 如果中文检索不到，尝试生成对应的英文专业术语关键词（例如：Spark Memory Management）。
        3. 尝试将问题拆解为更基础的词汇，或者换一个角度（例如：从底层原理、配置参数或架构图入手）。
        """
    
    prompt = f"""你是一个资深的搜索引擎优化专家和技术导师。
    你的目标是为用户问题生成最精准的检索关键词计划。

    【用户问题】：{user_query}
    {retry_instruction}

    【输出要求】：
    1. search_query: 必须是 1-3 个高度相关的关键词，适合向量数据库检索。
    2. reasoning: 简述你为什么要这样调整关键词。

    必须只输出 JSON 格式，例如：
    {{"search_query": "Spark 内存模型 架构", "reasoning": "由于之前的搜索偏离到了向量数据库，本次尝试使用更底层、包含架构特征的词汇。"}}
    """
    
    # 建议此处调用你新接入的 OpenAI 模型，因为这种反思逻辑对模型要求较高
    response = chat_completion(prompt, "You are a JSON engine. Stay creative in keyword generation.")
    
    try:
        plan_data = safe_parse_json(response)
        # 保持 state 中原有的 retry_count 不变，只更新 plan
        return {"plan": plan_data}
    except Exception as e:
        print(f"!!! [Planner] 格式解析失败: {e}")
        return {"plan": {"search_query": user_query, "reasoning": "解析失败，回退到原始问题"}}