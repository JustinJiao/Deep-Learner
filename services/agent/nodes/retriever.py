import time
import os
from services.agent.state import AgentState, AgentStep
from retrieval.pipeline import RetrievalPipeline
from llm.prompts import LOG_TEMPLATES

def retriever_node(state: AgentState):
    """
    检索节点：基于文件名去重的全局 ID 映射
    """
    print("--- [Node] 子任务检索与资料去重 ---")
    idx = state["current_step_idx"]
    step_task = state["plan"][idx]
    
    pipeline = RetrievalPipeline()
    search_results = pipeline.run(step_task)
    
    new_docs = []
    for res in search_results:
        full_path = res.metadata.get("source", "未知文档")
        file_name = os.path.basename(full_path)
        # 构造存入 context_pool 的内容
        new_docs.append({
            "id": file_name,
            "content": res.content
        })
    
    return {
        "context_pool": new_docs,
        "current_step_idx": idx + 1,
        "steps_log": [AgentStep(node="retriever", thought=f"子任务 {idx+1}{step_task} 检索并完成去重", timestamp=time.time())]
    }