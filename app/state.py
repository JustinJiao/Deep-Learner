from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    # 使用 Annotated[..., operator.add] 可以让消息自动追加
    messages: Annotated[List[str], operator.add]
    plan: str
    context: List[str]
    is_valid: bool
    retry_count: int