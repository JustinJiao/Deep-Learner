from typing import TypedDict, List, Annotated, Dict, Any
import operator

class AgentState(TypedDict):
    """
    Deep-Learner AI Agent 的状态定义
    """
    
    # 1. 对话历史：使用 Annotated 和 operator.add
    # 这确保了新消息会被追加到列表中，而不是覆盖旧消息
    messages: Annotated[List[str], operator.add]
    
    # 2. 检索计划：由 Planner 生成，存储结构化 JSON
    # 包含 {"search_query": "...", "reasoning": "..."}
    plan: Dict[str, Any]
    
    # 3. 知识库内容：由 Retriever 填充
    # 存储从 Milvus 检索回来的原始文本片段列表
    context: List[str]
    
    # 4. 教学回答：由 Tutor 生成，存储结构化 JSON
    # 包含 {"status": "success/no_data", "answer": "..."}
    answer: Dict[str, Any]
    
    # 5. 逻辑校验结果：由 Critic 填充
    # 决定了工作流是走向 END 还是回到 Planner 重试
    is_valid: bool
    
    # 6. 反馈建议：由 Critic 填充
    # 当校验失败时，记录具体原因，引导 Planner 在下一轮优化搜索词
    revision_notes: str
    
    # 7. 重试计数器
    # 防止 Agent 在无法解决的问题上陷入死循环（如知识库确实没数据时）
    retry_count: int