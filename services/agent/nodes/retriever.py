import time
import os
from services.agent.state import AgentState, AgentStep
from retrieval.pipeline import RetrievalPipeline
from config.settings import ResourceFactory
from llm.prompts import QUERY_REWRITE_SYSTEM_PROMPT, QUERY_REWRITE_HUMAN_PROMPT_TEMPLATE, LOG_TEMPLATES

def retriever_node(state: AgentState):
    """
    检索节点：执行混合检索，并建立 ID -> 文件名映射。
    """
    print("--- [Node] 执行检索节点 ---")
    llm_service = ResourceFactory.get_llm_service()
    pipeline = RetrievalPipeline()
    
    # 1. 查询重写
    query_to_use = state.get("query")
    if len(query_to_use) < 10:
        rewrite_prompt = QUERY_REWRITE_HUMAN_PROMPT_TEMPLATE.format(query=query_to_use)
        query_to_use = llm_service.chat_completion(
            prompt=rewrite_prompt, 
            system_prompt=QUERY_REWRITE_SYSTEM_PROMPT
        )
    
    # 2. 调用混合检索
    search_results = pipeline.run(query_to_use)
    
    # 3. 结构化封装结果与路径脱敏
    formatted_docs = []
    source_mapping = {}
    
    for i, res in enumerate(search_results):
        doc_id = i + 1
        # 获取纯净文件名（去掉冗长的本地路径）
        full_path = res.metadata.get("source", "未知文档")
        clean_name = os.path.basename(full_path) 
        
        formatted_docs.append({
            "id": doc_id,
            "content": res.content,
            "metadata": res.metadata # 内部仍保留完整元数据供前端使用
        })
        # 🌟 建立映射表存入 State
        source_mapping[doc_id] = clean_name
    
    log_text = LOG_TEMPLATES["retriever_success"].format(count=len(formatted_docs))
    
    return {
        "optimized_query": query_to_use,
        "retrieved_contents": formatted_docs,
        "source_mapping": source_mapping, # 🌟 更新状态
        "steps_log": [AgentStep(node="retriever", thought=log_text, timestamp=time.time())]
    }