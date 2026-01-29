import time
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from operator import add
from pydantic import BaseModel, Field

# 🌟 自定义合并函数：确保 source_mapping 在多轮检索中合并而不是覆盖
def merge_context_pool(existing: List[Dict], new: List[Dict]) -> List[Dict]:
    """
    🌟 核心修复：直接基于文件名(id)进行去重
    """
    # 以文件名作为 key 进行去重
    merged = {doc['id']: doc for doc in existing}
    for doc in new:
        merged[doc['id']] = doc
    return list(merged.values())

class AgentStep(BaseModel):
    node: str = Field(..., description="节点名称")
    thought: str = Field(..., description="思考过程")
    timestamp: float = Field(default_factory=time.time)

class AgentState(TypedDict):
    query: str                       # 原始用户输入
    rewritten_query: str             # 对齐后的标准意图
    plan: List[str]                  # Planner 拆解的子任务列表
    current_step_idx: int            # 迭代检索的当前索引
    
    # 使用 Annotated 确保增量更新
    context_pool: Annotated[List[Dict[str, Any]], merge_context_pool] 
    # source_mapping: Annotated[Dict[str, str], merge_mapping] 
    
    response: str                    # Tutor 生成的回答
    critique: Optional[str]          # Critic 的反馈
    is_hallucination: bool           # 幻觉标记
    loop_count: int                  # 整体重试计数
    
    steps_log: Annotated[List[AgentStep], add]
    messages: Annotated[List[Any], add]