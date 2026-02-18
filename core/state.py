# core/state.py

import time
from typing import List, Dict, Any, Optional, Literal, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from operator import add
from core.plan import ExecutionPlan


class StepLog(BaseModel):
    node: str
    info: str
    timestamp: float = Field(default_factory=time.time)


class Intent(TypedDict):
    type: Literal["chat", "research"]
    confidence: float


class STMState(TypedDict):
    summary: List[str]
    messages: List[Dict[str, Any]]
    recent_messages: List[Dict[str, Any]]
    compressed_until: int


class AgentState(TypedDict, total=False):
    # 输入
    query: str
    session_id: str

    # Memory 投影字段（给 LLM）
    short_term_memory: str
    recent_messages: List[Dict[str, Any]]
    long_term_memory: str

    # 控制流
    intent: Intent
    is_direct_path: bool
    plan: ExecutionPlan
    loop_count: int

    # RAG
    rewritten_query: str
    context_pool: List[Dict[str, Any]]

    # 输出
    response: str

    # Verify / Repair
    is_hallucination: bool
    critique: Dict[str, Any]  # VerifyPrompt 输出（包含 error_type/next_step/critique）
    repair_hint: str

    # Debug
    steps_log: Annotated[List[StepLog], add]

    # 内部字段
    _stm_to_compress: List[Dict[str, Any]]
