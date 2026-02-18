# scripts/system_test.py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

"""
Deep-Learner SYSTEM TEST 2.0

新增验证：
- Memory 跨轮使用
- Summary 参与 compose
- 失败自动 dump STM
"""

import sys
from core.executor import AgentExecutor
from session.store import get_session
from memory.ltm import LTM


def fail_dump(session_id, state, message):
    print(f"\n❌ FAIL: {message}")

    ctx = get_session(session_id)
    stm = ctx.stm

    print("\n===== DEBUG DUMP =====")
    print("Response:", state.get("response"))
    print("\nRecent Messages:", state.get("recent_messages"))
    print("\nShort Term Memory:", state.get("short_term_memory"))
    print("\nSummary Blocks:", stm.get("summary"))
    print("\nCompressed Until:", stm.get("compressed_until"))
    print("\nAll Messages:", stm.get("messages"))
    print("======================\n")

    sys.exit(1)


def assert_true(cond, msg, session_id=None, state=None):
    if not cond:
        if session_id and state:
            fail_dump(session_id, state, msg)
        else:
            print(f"❌ FAIL: {msg}")
            sys.exit(1)
    else:
        print(f"✅ PASS: {msg}")


def run_test():
    print("\n==============================")
    print("Deep-Learner SYSTEM TEST 2.0 START")
    print("==============================\n")

    executor = AgentExecutor()
    session_id = "system-test-session"

    test_queries = [
        "hi",
        "我喜欢蓝色",
        "我住在北京",
        "transformer是什么？",
        "再讲详细一点",
        "总结一下我们刚才说的",
        "我刚才说我喜欢什么颜色？",
        "我刚才说我住在哪？",
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n---- Round {i} ----")

        state = executor.run(session_id=session_id, query=q)

        assert_true(
            state.get("run_status") == "ok",
            "run_status should be ok",
            session_id,
            state
        )

        assert_true(
            state.get("response") is not None,
            "response exists",
            session_id,
            state
        )

        assert_true(
            "intent" in state,
            "intent exists",
            session_id,
            state
        )

        assert_true(
            "plan" in state,
            "plan exists",
            session_id,
            state
        )

        plan = state["plan"]
        assert_true(
            plan.is_finished(),
            "plan finished",
            session_id,
            state
        )

    # --------------------------
    # Memory 行为验证
    # --------------------------

    ctx = get_session(session_id)
    stm = ctx.stm

    messages = stm.get("messages", [])
    recent = stm.get("recent_messages", [])
    summary = stm.get("summary", [])
    compressed_until = stm.get("compressed_until")

    print("\nChecking STM structure...")

    assert_true(
        len(messages) == len(test_queries),
        "messages count equals number of queries"
    )

    assert_true(
        len(recent) <= 3,
        "recent_messages window correct"
    )

    assert_true(
        isinstance(summary, list),
        "summary is list"
    )

    assert_true(
        compressed_until <= len(messages),
        "compressed_until valid"
    )

    # 如果超过5轮必须压缩
    if len(messages) >= 5:
        assert_true(
            len(summary) >= 1,
            "summary compressed at least once"
        )

    # --------------------------
    # Memory Recall 验证
    # --------------------------

    print("\nChecking memory recall...")

    # 颜色记忆
    state_color = executor.run(
        session_id=session_id,
        query="我刚才说我喜欢什么颜色？"
    )

    assert_true(
        "蓝色" in state_color.get("response", ""),
        "memory recall correct for 蓝色",
        session_id,
        state_color
    )

    # 地点记忆（测试 summary 是否参与）
    state_city = executor.run(
        session_id=session_id,
        query="我刚才说我住在哪？"
    )

    assert_true(
        "北京" in state_city.get("response", ""),
        "memory recall correct for 北京",
        session_id,
        state_city
    )

    # --------------------------
    # LTM 验证
    # --------------------------

    print("\nChecking LTM...")

    ltm = LTM()
    col = ltm.collection

    results = col.query(
        expr="key != ''",
        output_fields=["key", "content"],
        limit=100,
    )

    assert_true(
        len(results) >= 1,
        "LTM stored at least one fact"
    )

    print("\n==============================")
    print("🎉 ALL SYSTEM TESTS PASSED")
    print("==============================\n")


if __name__ == "__main__":
    run_test()
