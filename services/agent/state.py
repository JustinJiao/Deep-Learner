from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from operator import add
from pydantic import BaseModel, Field
import time

# 定义单个节点的思考快照，用于 WebSocket 实时展示思考链
class AgentStep(BaseModel):
    node: str = Field(..., description="执行节点")
    thought: str = Field(..., description="节点思考/执行逻辑")
    timestamp: float = Field(default_factory=time.time)

class AgentState(TypedDict):
    """
    Deep-Learner 状态协议：
    支持认知闭环、状态累加、以及后期微调所需的数据记录。
    """
    # --- 原始输入与查询优化 ---
    query: str                                  # 用户的原始问题
    optimized_query: Optional[str]              # 经过重写后的更利于检索的问题
    
    # --- 任务规划 ---
    plan: List[str]                             # Planner 拆解的子任务清单
    
    # --- 知识内容 (核心解耦点) ---
    # 存储检索到的 SearchResult 字典列表，方便后续引用溯源
    retrieved_contents: List[Dict[str, Any]]    
    
    # --- 生成结果与质量评估 ---
    response: str                               # Tutor 生成的教学答案
    critique: Optional[str]                     # Critic 给出的逻辑审核意见
    is_hallucination: bool                      # 幻觉/逻辑错误检测结果
    
    # --- 记忆与日志 (使用 Annotated[..., add] 实现增量更新) ---
    # 每次节点运行产生的新消息会自动附加，而不是覆盖旧数据
    steps_log: Annotated[List[AgentStep], add]  
    messages: Annotated[List[Any], add]         # 对话历史，兼容 ChatMessage 格式
    
    # --- 性能元数据 ---
    metadata: Dict[str, Any]                    # 存储 Token 消耗、总耗时等