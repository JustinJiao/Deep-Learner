import time
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from operator import add
from pydantic import BaseModel, Field

# ==========================================
# 1. 原子思考单元：用于全链路可观察性 (Observability)
# ==========================================
class AgentStep(BaseModel):
    """
    记录单个节点的执行痕迹。
    亮点：支持 LangSmith 追踪与前端思考链展示。
    """
    node: str = Field(..., description="当前执行任务的节点名称")
    thought: str = Field(..., description="智能体的思考过程/中间逻辑")
    timestamp: float = Field(default_factory=time.time, description="执行时间戳")

# ==========================================
# 2. 核心状态协议：驱动全流程的数据载体
# ==========================================
class AgentState(TypedDict):
    # --- [基础输入层] 所有节点共享 ---
    query: str  # 用户原始输入
    
    # --- [Planner 节点专用] 任务编排 ---
    # 作用：将复杂问题拆解为可执行的子目标
    plan: List[str] 
    current_step_idx: int # 记录当前执行到计划的第几步
    
    # --- [Retriever 节点专用] 知识召回与脱敏 ---
    # 作用：实现物理路径与逻辑 ID 的隔离，保护服务器隐私
    optimized_query: Optional[str]              # 经过 LLM 重写后的检索词
    retrieved_contents: List[Dict[str, Any]]    # 检索出的纯净正文块 (含逻辑 ID)
    source_mapping: Dict[int, str]              # 🌟 核心亮点：ID -> 原始文件名映射表
    
    # --- [Tutor 节点专用] 教学生成 ---
    # 作用：基于检索内容生成带引用的启发式回答
    response: str # 最终生成的导师回答内容
    
    # --- [Critic 节点专用] 逻辑审计 ---
    # 作用：进行幻觉检测与逻辑一致性校验，决定是否回炉重造
    critique: Optional[str]   # 审计意见 (PASS/FAIL)
    is_hallucination: bool    # 幻觉标记位
    
    # --- [全局增量层] 使用 Annotated[..., add] 实现非覆盖式更新 ---
    # 作用：记录 Agent 的生命周期全过程
    steps_log: Annotated[List[AgentStep], add]  # 思考链日志 (按顺序累加)
    messages: Annotated[List[Any], add]         # 聊天历史 (支持多轮对话上下文)
    
    # --- [扩展层] 基础设施配置 ---
    metadata: Dict[str, Any] # 存储如 session_id, model_version 等元数据