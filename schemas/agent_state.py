from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from operator import add
from pydantic import BaseModel, Field

# 1. 定义原子思考单元（用于前端 WebSocket 可视化）
class AgentStep(BaseModel):
    node: str = Field(..., description="当前执行任务的节点名称")
    thought: str = Field(..., description="智能体的思考过程/中间逻辑")
    timestamp: float = Field(..., description="执行时间戳")

# 2. 定义核心状态协议
class AgentState(TypedDict):
    """
    Deep-Learner 工业级状态协议
    支持：状态累加、断点续传、思考链追踪、多路检索结果存储
    """
    # --- 输入与意图 ---
    query: str                                  # 用户原始输入
    optimized_query: Optional[str]              # 经过查询重写（Query Rewriter）后的检索语句
    
    # --- 规划与任务 ---
    plan: List[str]                             # Planner 节点拆解的任务清单
    current_step_idx: int                       # 当前执行到计划的第几步
    
    # --- 知识内容 ---
    # 使用你 retrieval 模块返回的 SearchResult 字典列表
    retrieved_contents: List[Dict[str, Any]]    # 混合检索后的原始文档片段与元数据
    
    # --- 生成与评估 ---
    response: str                               # Tutor 节点生成的教学内容
    critique: Optional[str]                     # Critic 节点的逻辑校验反馈
    is_hallucination: bool                      # 幻觉检测标志位
    
    # --- 记忆与日志 (关键：使用 Annotated[..., add] 实现增量更新) ---
    # 每经过一个节点，新的日志会自动附加到列表末尾，而不是覆盖旧日志
    steps_log: Annotated[List[AgentStep], add]  # 供 WebSocket 实时推送的决策链路
    
    # 对话历史，兼容 LangChain 的 Message 格式，方便接入 Redis 记忆
    messages: Annotated[List[Any], add]         
    
    # --- 工业级元数据 ---
    # 用于监控：Token 消耗、各节点耗时、Trace ID 等
    metadata: Dict[str, Any]