# core/errors.py

class AgentError(Exception):
    """Agent 运行期基础异常"""


class PromptContractError(ValueError):
    """Prompt 违反 READ / WRITE 约束"""


class InvalidPlanError(AgentError):
    """Plan 结构非法 / Graph 转移非法"""


class MaxLoopExceededError(AgentError):
    """超过最大修复次数"""


class UnknownNodeError(AgentError):
    """Plan 中出现未注册节点"""
