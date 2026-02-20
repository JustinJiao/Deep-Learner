import core.executor as executor_mod
from core.executor import AgentExecutor
from core.plan import ExecutionPlan


def test_executor_full_run_smoke(monkeypatch):
    def stm_read_node(state):
        return state

    def intent_node(state):
        state["intent"] = {"type": "chat", "confidence": 1.0}
        return state

    def planner_node(state):
        state["plan"] = ExecutionPlan(steps=["compose"], max_loops=1)
        return state

    def compose_node(state):
        state["response"] = "stub-response"
        return state

    def finalize_node(state):
        state["finalized"] = True
        return state

    def stm_write_node(state):
        state["stm_written"] = True
        return state

    def stm_summary_node(state):
        state["stm_summarized"] = True
        return state

    def persist_ltm_node(state):
        state["ltm_persisted"] = True
        return state

    fake_registry = {
        "stm_read": stm_read_node,
        "intent": intent_node,
        "planner": planner_node,
        "compose": compose_node,
        "finalize": finalize_node,
        "stm_write": stm_write_node,
        "stm_summary": stm_summary_node,
        "persist_ltm": persist_ltm_node,
    }

    monkeypatch.setattr(executor_mod, "NODE_REGISTRY", fake_registry)

    final_state = AgentExecutor().run(
        session_id="test-session",
        query="Spark 内存调优有哪些关键参数？",
    )

    assert final_state["run_status"] == "ok"
    assert final_state["response"] == "stub-response"
    assert final_state["finalized"] is True
    assert final_state["stm_written"] is True
    assert final_state["stm_summarized"] is True
    assert final_state["ltm_persisted"] is True
