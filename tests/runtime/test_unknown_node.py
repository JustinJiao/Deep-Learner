import pytest
from core.executor import AgentExecutor
from core.plan import ExecutionPlan
from core.errors import UnknownNodeError


def intent_node(state):
    state["intent"] = {"type": "research", "confidence": 1.0}
    return state


def planner_node(state):
    state["plan"] = ExecutionPlan(["not_exist_node"])
    return state


REGISTRY = {
    "intent": intent_node,
    "planner": planner_node,
}


def test_unknown_node():
    executor = AgentExecutor(REGISTRY)

    with pytest.raises(UnknownNodeError):
        executor.run({"query": "test"})
