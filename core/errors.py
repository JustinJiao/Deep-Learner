# core/errors.py

class AgentError (Exception ):
    """Basic exception during Agent runtime"""


class PromptContractError (ValueError ):
    """Prompt violates READ / WRITE constraints"""


class InvalidPlanError (AgentError ):
    """Plan structure is illegal/Graph transfer is illegal"""


class MaxLoopExceededError (AgentError ):
    """Exceeded maximum number of repairs"""


class UnknownNodeError (AgentError ):
    """Unregistered nodes appear in Plan"""


class InvalidStateTransitionError (AgentError ):
    """An illegal transition occurred in the runtime state machine."""


class RuntimeMaxTransitionExceededError (AgentError ):
    """The number of runtime state transitions exceeds the upper limit"""


class NodeContractViolationError (AgentError ):
    """Node read and write contract violation"""
