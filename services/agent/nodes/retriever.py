# services/agent/nodes/retriever.py
from services.agent.state import AgentState, AgentStep
from retrieval.pipeline import RetrievalPipeline
from config.settings import ResourceFactory
from llm.prompts import QUERY_REWRITE_SYSTEM_PROMPT, QUERY_REWRITE_HUMAN_PROMPT_TEMPLATE, LOG_TEMPLATES

def retriever_node(state: AgentState):
    """
    检索节点：执行混合检索，并可选执行查询重写
    """
    print("--- [Node] 执行检索节点 ---")
    
    llm_service = ResourceFactory.get_llm_service()
    pipeline = RetrievalPipeline()
    
    # 1. 工业级增强：如果原始 Query 太短，先通过 LLM 重写
    query_to_use = state.get("query")
    if len(query_to_use) < 10: # 假设短查询需要重写
        rewrite_prompt = QUERY_REWRITE_HUMAN_PROMPT_TEMPLATE.format(query=query_to_use)
        query_to_use = llm_service.chat_completion(
            prompt=rewrite_prompt, 
            system_prompt=QUERY_REWRITE_SYSTEM_PROMPT
        )
    
    # 2. 调用混合检索流水线
    search_results = pipeline.run(query_to_use)
    
    # 3. 封装结果
    formatted_docs = [
        {
            "id": res.id,
            "content": res.content,
            "metadata": res.metadata
        } for res in search_results
    ]
    
    # 4. 使用解耦的日志模版
    log_text = LOG_TEMPLATES["retriever_success"].format(count=len(formatted_docs))
    
    return {
        "optimized_query": query_to_use,
        "retrieved_contents": formatted_docs,
        "steps_log": [AgentStep(node="retriever", thought=log_text)]
    }