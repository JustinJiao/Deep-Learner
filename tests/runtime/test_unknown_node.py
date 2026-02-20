import core.executor as executor_mod
from core.executor import AgentExecutor
from core.plan import ExecutionPlan


def test_unknown_node_sets_error_status(monkeypatch):
    def stm_read_node(state):
        return state

    def intent_node(state):
        state["intent"] = {"type": "research", "confidence": 1.0}
        return state

    def planner_node(state):
        state["plan"] = ExecutionPlan(["not_exist_node"])
        return state

    def finalize_node(state):
        return state

    def stm_write_node(state):
        return state

    def stm_summary_node(state):
        return state

    def persist_ltm_node(state):
        return state

    fake_registry = {
        "stm_read": stm_read_node,
        "intent": intent_node,
        "planner": planner_node,
        "finalize": finalize_node,
        "stm_write": stm_write_node,
        "stm_summary": stm_summary_node,
        "persist_ltm": persist_ltm_node,
    }

    monkeypatch.setattr(executor_mod, "NODE_REGISTRY", fake_registry)

    state = AgentExecutor().run(session_id="test-session", query="test")

    assert state["run_status"] == "error"
    assert state["error"]["type"] == "UnknownNodeError"
