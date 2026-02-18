# tests/runtime/test_executor_smoke.py
from core.executor import AgentExecutor

def test_executor_full_run_smoke():
    executor = AgentExecutor()

    state = {
        "query": "Spark 内存调优有哪些关键参数？",
        "messages": [],
    }

    final_state = executor.run(state)

    # 1️⃣ 核心输出存在
    assert "response" in final_state
    assert isinstance(final_state["response"], str)
    assert len(final_state["response"]) > 20

    # 2️⃣ 至少走过 compose
    nodes = [s.node for s in final_state.get("steps_log", [])]
    assert "compose" in nodes

    # 3️⃣ 没有无限循环
    assert final_state.get("loop_count", 0) <= 3
