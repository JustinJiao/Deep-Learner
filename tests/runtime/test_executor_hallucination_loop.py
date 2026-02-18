from core.executor import AgentExecutor
from core.plan import ExecutionPlan


def intent_node(state):
    state["intent"] = {"type": "research", "confidence": 1.0}
    return state


def planner_node(state):
    state["plan"] = ExecutionPlan(
        steps=["compose", "verify", "finalize"],
        max_loops=2,
    )
    state["loop_count"] = 0
    return state


def compose_node(state):
    state["response"] = "答案"
    return state


def verify_node(state):
    if state["loop_count"] == 0:
        state["is_hallucination"] = True
        state["critique"] = "FAIL"
    else:
        state["is_hallucination"] = False
    return state


def repair_node(state):
    state["loop_count"] += 1
    state["plan"].jump_to("compose")
    return state


def finalize_node(state):
    return state


def memory_write_node(state):
    return state


REGISTRY = {
    "intent": intent_node,
    "planner": planner_node,
    "compose": compose_node,
    "verify": verify_node,
    "repair": repair_node,
    "finalize": finalize_node,
    "memory_write": memory_write_node,
}


def test_hallucination_repair_loop():
    executor = AgentExecutor(REGISTRY)

    state = executor.run({"query": "test"})

    assert state["loop_count"] == 1
    assert state["response"] == "答案"
