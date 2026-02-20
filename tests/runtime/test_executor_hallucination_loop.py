import core.executor as executor_mod
from core.executor import AgentExecutor
from core.plan import ExecutionPlan


def test_hallucination_repair_loop(monkeypatch):
    def stm_read_node(state):
        return state

    def intent_node(state):
        state["intent"] = {"type": "research", "confidence": 1.0}
        return state

    def planner_node(state):
        state["plan"] = ExecutionPlan(
            steps=["compose", "verify"],
            max_loops=2,
        )
        return state

    def compose_node(state):
        state["compose_calls"] = state.get("compose_calls", 0) + 1
        state["response"] = f"answer-{state['compose_calls']}"
        return state

    def verify_node(state):
        verify_calls = state.get("verify_calls", 0) + 1
        state["verify_calls"] = verify_calls
        state["is_hallucination"] = verify_calls == 1
        return state

    def repair_node(state):
        state["plan"].jump_to("compose")
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
        "compose": compose_node,
        "verify": verify_node,
        "repair": repair_node,
        "finalize": finalize_node,
        "stm_write": stm_write_node,
        "stm_summary": stm_summary_node,
        "persist_ltm": persist_ltm_node,
    }

    monkeypatch.setattr(executor_mod, "NODE_REGISTRY", fake_registry)

    state = AgentExecutor().run(session_id="test-session", query="test")

    assert state["run_status"] == "ok"
    assert state["loop_count"] == 1
    assert state["compose_calls"] == 2
    assert state["verify_calls"] == 2
    assert state["response"] == "answer-2"
