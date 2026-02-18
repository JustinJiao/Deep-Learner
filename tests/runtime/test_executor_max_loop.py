import pytest
from core.executor import AgentExecutor
from core.plan import ExecutionPlan
from core.errors import MaxLoopExceededError


def intent_node(state):
    state["intent"] = {"type": "research", "confidence": 1.0}
    return state


def planner_node(state):
    state["plan"] = ExecutionPlan(
        steps=["verify"],
        max_loops=1,
    )
    state["loop_count"] = 0
    return state


def verify_node(state):
    state["is_hallucination"] = True
    return state


def repair_node(state):
    state["loop_count"] += 1
    if state["loop_count"] > state["plan"].max_loops:
        raise MaxLoopExceededError("Too many retries")
    return state


def finalize_node(state):
    return state


def memory_write_node(state):
    return state


REGISTRY = {
    "intent": intent_node,
    "planner": planner_node,
    "verify": verify_node,
    "repair": repair_node,
    "finalize": finalize_node,
    "memory_write": memory_write_node,
}


def test_max_loop_exceeded():
    executor = AgentExecutor(REGISTRY)

    with pytest.raises(MaxLoopExceededError):
        executor.run({"query": "test"})
