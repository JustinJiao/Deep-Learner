from core.executor import AgentExecutor
from config.settings import AppConfig
from session import store as session_store
from tools.retrieve_tool.base import SearchResult


def _clear_session_store():
    session_store._SESSION_STORE.clear()


def test_full_chain_happy_path_step_logs(monkeypatch):
    _clear_session_store()
    monkeypatch.setattr(AppConfig, "LTM_WRITE_ENABLED", True)

    calls = {
        "retrieve": 0,
        "compose": 0,
        "verify": 0,
        "persist_entries": [],
    }

    class FakeRecallLTM:
        def recall(self, query):
            assert query == "Explain transformer"
            return ["ltm-m1", "ltm-m2"]

    class FakePipeline:
        def run(self, query):
            calls["retrieve"] += 1
            assert query == "transformer architecture details"
            return [
                SearchResult(
                    id="doc-1",
                    content="transformer uses self-attention",
                    score=0.9,
                    metadata={"title": "vector-doc"},
                    source_type="vector",
                ),
                SearchResult(
                    id="doc-2",
                    content="multi-head attention enables parallel views",
                    score=0.8,
                    metadata={"title": "keyword-doc"},
                    source_type="keyword",
                ),
            ]

    class FakePersistLTM:
        def upsert(self, entries):
            calls["persist_entries"] = entries
            return len(entries)

    def fake_intent_run_prompt(prompt_cls, state):
        return {"intent": {"type": "research", "confidence": 0.99}}

    def fake_rewrite_run_prompt(prompt_cls, state):
        return {"rewritten_query": "transformer architecture details"}

    def fake_compose_run_prompt(prompt_cls, state):
        calls["compose"] += 1
        return {
            "response": "Transformer uses attention layers.",
            "citations": [
                {
                    "id": "doc-1",
                    "title": "vector-doc",
                    "score": 0.9,
                    "quote": "transformer uses self-attention",
                }
            ],
        }

    def fake_verify_run_prompt(prompt_cls, state):
        calls["verify"] += 1
        # Intentionally inconsistent PASS payload; verify node should normalize it.
        return {
            "is_hallucination": False,
            "error_type": "retrieval_insufficient",
            "next_step": "compose",
            "critique": "answer is grounded",
        }

    def fake_persist_run_prompt(prompt_cls, state):
        return {
            "fact_candidates": [
                {"key": "k1", "type": "fact", "content": "c1", "score": 0.95},
                {"key": "k2", "type": "fact", "content": "c2", "score": 0.2},
            ]
        }

    monkeypatch.setattr("nodes.recall_ltm.LTM", FakeRecallLTM)
    monkeypatch.setattr("nodes.retrieve.RetrievalPipeline", FakePipeline)
    monkeypatch.setattr("nodes.persist_ltm.LTM", FakePersistLTM)
    monkeypatch.setattr("nodes.intent.run_prompt", fake_intent_run_prompt)
    monkeypatch.setattr("nodes.query_rewrite.run_prompt", fake_rewrite_run_prompt)
    monkeypatch.setattr("nodes.compose.run_prompt", fake_compose_run_prompt)
    monkeypatch.setattr("nodes.verify.run_prompt", fake_verify_run_prompt)
    monkeypatch.setattr("nodes.persist_ltm.run_prompt", fake_persist_run_prompt)

    state = AgentExecutor().run(session_id="e2e-session-1", query="Explain transformer")

    assert state["run_status"] == "ok"
    assert state["loop_count"] == 0
    assert state["long_term_memory"] == "ltm-m1 | ltm-m2"
    assert calls["retrieve"] == 1
    assert calls["compose"] == 1
    assert calls["verify"] == 1
    assert len(calls["persist_entries"]) == 1
    assert calls["persist_entries"][0]["key"] == "k1"

    nodes = [log.node for log in state["steps_log"]]
    assert nodes == [
        "stm_read",
        "intent",
        "planner",
        "recall_ltm",
        "query_rewrite",
        "retrieve",
        "compose",
        "verify",
        "finalize",
        "stm_write",
        "stm_summary",
        "persist_ltm",
    ]

    verify_log = next(log for log in state["steps_log"] if log.node == "verify")
    assert verify_log.info["llm_output"]["verdict"] == "PASS"
    assert verify_log.info["llm_output"]["error_type"] == ""
    assert verify_log.info["llm_output"]["next_step"] == ""

    retrieve_log = next(log for log in state["steps_log"] if log.node == "retrieve")
    assert retrieve_log.info["state"]["used_rewritten_query"] is True
    assert retrieve_log.info["memory"]["context_pool_count"] == 2

    persist_log = next(log for log in state["steps_log"] if log.node == "persist_ltm")
    assert persist_log.info["llm_output"]["fact_candidates_raw_count"] == 2
    assert persist_log.info["llm_output"]["fact_candidates_kept_count"] == 1
    assert persist_log.info["memory"]["stored_count"] == 1


def test_full_chain_repair_then_pass_step_logs(monkeypatch):
    _clear_session_store()
    monkeypatch.setattr(AppConfig, "LTM_WRITE_ENABLED", True)

    calls = {
        "retrieve": 0,
        "compose": 0,
        "verify": 0,
    }

    class FakeRecallLTM:
        def recall(self, query):
            return ["ltm-x"]

    class FakePipeline:
        def run(self, query):
            calls["retrieve"] += 1
            if calls["retrieve"] == 1:
                return [
                    SearchResult(
                        id="doc-r1",
                        content="weak context",
                        score=0.3,
                        metadata={"title": "first-recall"},
                        source_type="vector",
                    )
                ]
            return [
                SearchResult(
                    id="doc-r2",
                    content="strong context",
                    score=0.95,
                    metadata={"title": "second-recall"},
                    source_type="vector",
                )
            ]

    class FakePersistLTM:
        def upsert(self, entries):
            return len(entries)

    def fake_intent_run_prompt(prompt_cls, state):
        return {"intent": {"type": "research", "confidence": 0.99}}

    def fake_rewrite_run_prompt(prompt_cls, state):
        return {"rewritten_query": "rewritten q"}

    def fake_compose_run_prompt(prompt_cls, state):
        calls["compose"] += 1
        top_doc = state.get("context_pool", [{}])[0].get("id", "")
        return {
            "response": f"answer from {top_doc}",
            "citations": [],
        }

    def fake_verify_run_prompt(prompt_cls, state):
        calls["verify"] += 1
        if calls["verify"] == 1:
            return {
                "is_hallucination": True,
                "error_type": "retrieval_insufficient",
                "next_step": "retrieve",
                "critique": "need better context",
            }
        return {
            "is_hallucination": False,
            "error_type": "retrieval_insufficient",
            "next_step": "compose",
            "critique": "now grounded",
        }

    def fake_persist_run_prompt(prompt_cls, state):
        return {
            "fact_candidates": [
                {"key": "k", "type": "fact", "content": "fact", "score": 0.9}
            ]
        }

    monkeypatch.setattr("nodes.recall_ltm.LTM", FakeRecallLTM)
    monkeypatch.setattr("nodes.retrieve.RetrievalPipeline", FakePipeline)
    monkeypatch.setattr("nodes.persist_ltm.LTM", FakePersistLTM)
    monkeypatch.setattr("nodes.intent.run_prompt", fake_intent_run_prompt)
    monkeypatch.setattr("nodes.query_rewrite.run_prompt", fake_rewrite_run_prompt)
    monkeypatch.setattr("nodes.compose.run_prompt", fake_compose_run_prompt)
    monkeypatch.setattr("nodes.verify.run_prompt", fake_verify_run_prompt)
    monkeypatch.setattr("nodes.persist_ltm.run_prompt", fake_persist_run_prompt)

    state = AgentExecutor().run(session_id="e2e-session-2", query="Explain transformer")

    assert state["run_status"] == "ok"
    assert state["loop_count"] == 1
    assert calls["retrieve"] == 2
    assert calls["compose"] == 2
    assert calls["verify"] == 2
    assert state["response"] == "answer from doc-r2"
    assert state["context_pool"][0]["id"] == "doc-r2"

    nodes = [log.node for log in state["steps_log"]]
    assert nodes == [
        "stm_read",
        "intent",
        "planner",
        "recall_ltm",
        "query_rewrite",
        "retrieve",
        "compose",
        "verify",
        "executor",
        "repair",
        "retrieve",
        "compose",
        "verify",
        "finalize",
        "stm_write",
        "stm_summary",
        "persist_ltm",
    ]

    repair_log = next(log for log in state["steps_log"] if log.node == "repair")
    assert repair_log.info["decision"]["next_step_requested"] == "retrieve"
    assert repair_log.info["decision"]["next_step_applied"] == "retrieve"
    assert repair_log.info["decision"]["context_pool_cleared"] is True

    verify_logs = [log for log in state["steps_log"] if log.node == "verify"]
    assert verify_logs[0].info["llm_output"]["verdict"] == "FAIL"
    assert verify_logs[0].info["llm_output"]["error_type"] == "retrieval_insufficient"
    assert verify_logs[1].info["llm_output"]["verdict"] == "PASS"
    assert verify_logs[1].info["llm_output"]["error_type"] == ""
    assert verify_logs[1].info["llm_output"]["next_step"] == ""
