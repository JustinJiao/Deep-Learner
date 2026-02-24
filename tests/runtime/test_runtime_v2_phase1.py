import core.executor as executor_mod
from config.settings import AppConfig
from core.executor import AgentExecutor


def _base_tail_nodes(calls):
    def finalize_node(state):
        calls.append("finalize")
        return state

    def stm_write_node(state):
        calls.append("stm_write")
        return state

    def stm_summary_node(state):
        calls.append("stm_summary")
        return state

    def persist_ltm_node(state):
        calls.append("persist_ltm")
        return state

    return {
        "finalize": finalize_node,
        "stm_write": stm_write_node,
        "stm_summary": stm_summary_node,
        "persist_ltm": persist_ltm_node,
    }


def test_runtime_v2_memory_sufficient_skips_phase1(monkeypatch):
    calls = []

    def stm_read_node(state):
        calls.append("stm_read")
        state["short_term_memory"] = "stm"
        state["recent_messages"] = []
        return state

    def resolve_query_reference_node(state):
        calls.append("resolve_query_reference")
        state["resolved_query"] = state.get("query", "")
        return state

    def ltm_recall_node(state):
        calls.append("ltm_recall")
        state["long_term_memory"] = "ltm"
        return state

    def compose_memory_draft_node(state):
        calls.append("compose_memory_draft")
        state["draft_answer"] = "answer-from-memory"
        state["draft_confidence"] = 0.9
        state["used_memory_chunks"] = 2
        return state

    def verify_memory_node(state):
        calls.append("verify_memory")
        state["memory_verdict"] = "SUFFICIENT"
        state["memory_reason"] = "enough memory"
        state["memory_risk_level"] = "LOW"
        return state

    fake_registry = {
        "stm_read": stm_read_node,
        "resolve_query_reference": resolve_query_reference_node,
        "ltm_recall": ltm_recall_node,
        "compose_memory_draft": compose_memory_draft_node,
        "verify_memory": verify_memory_node,
        **_base_tail_nodes(calls),
    }

    monkeypatch.setattr(executor_mod, "NODE_REGISTRY", fake_registry)
    monkeypatch.setattr(AppConfig, "RUNTIME_V2_ENABLED", True)

    out = AgentExecutor().run(session_id="v2-s1", query="q")

    assert out["run_status"] == "ok"
    assert out["response"] == "answer-from-memory"
    assert out["runtime_stage"] == "FINALIZE"
    assert out["transition_count"] == 1
    assert calls == [
        "stm_read",
        "resolve_query_reference",
        "ltm_recall",
        "compose_memory_draft",
        "verify_memory",
        "finalize",
        "stm_write",
        "stm_summary",
        "persist_ltm",
    ]


def test_runtime_v2_phase1_pass(monkeypatch):
    calls = []

    def stm_read_node(state):
        calls.append("stm_read")
        state["short_term_memory"] = "stm"
        state["recent_messages"] = []
        return state

    def resolve_query_reference_node(state):
        calls.append("resolve_query_reference")
        state["resolved_query"] = state.get("query", "")
        return state

    def ltm_recall_node(state):
        calls.append("ltm_recall")
        state["long_term_memory"] = "ltm"
        return state

    def compose_memory_draft_node(state):
        calls.append("compose_memory_draft")
        state["draft_answer"] = "weak-memory"
        state["draft_confidence"] = 0.2
        state["used_memory_chunks"] = 0
        return state

    def verify_memory_node(state):
        calls.append("verify_memory")
        state["memory_verdict"] = "NEED_RETRIEVE"
        state["memory_reason"] = "memory not enough"
        state["memory_risk_level"] = "HIGH"
        return state

    def rewrite_query_for_retrieval_node(state):
        calls.append("rewrite_query_for_retrieval")
        state["retrieval_query"] = state.get("resolved_query", state.get("query", ""))
        return state

    def retrieve_phase1_node(state):
        calls.append("retrieve_phase1")
        state["phase1_candidates"] = [{"id": "d1"}]
        return state

    def rerank_phase1_node(state):
        calls.append("rerank_phase1")
        state["phase1_reranked"] = [{"id": "d1"}]
        state["context_pool"] = [{"id": "d1"}]
        return state

    def compose_with_context_node(state):
        calls.append("compose_with_context")
        state["response"] = "answer-with-context"
        state["citations"] = [{"id": "d1"}]
        state["response_revision"] = state.get("response_revision", 0) + 1
        return state

    def strict_verify_node(state):
        calls.append("strict_verify")
        state["strict_verdict"] = "PASS"
        state["strict_reason"] = "grounded"
        state["verified_revision"] = state.get("response_revision", 0)
        return state

    fake_registry = {
        "stm_read": stm_read_node,
        "resolve_query_reference": resolve_query_reference_node,
        "ltm_recall": ltm_recall_node,
        "compose_memory_draft": compose_memory_draft_node,
        "verify_memory": verify_memory_node,
        "rewrite_query_for_retrieval": rewrite_query_for_retrieval_node,
        "retrieve_phase1": retrieve_phase1_node,
        "rerank_phase1": rerank_phase1_node,
        "compose_with_context": compose_with_context_node,
        "strict_verify": strict_verify_node,
        **_base_tail_nodes(calls),
    }

    monkeypatch.setattr(executor_mod, "NODE_REGISTRY", fake_registry)
    monkeypatch.setattr(AppConfig, "RUNTIME_V2_ENABLED", True)

    out = AgentExecutor().run(session_id="v2-s2", query="q")

    assert out["run_status"] == "ok"
    assert out["response"] == "answer-with-context"
    assert out["runtime_stage"] == "FINALIZE"
    assert out["transition_count"] == 2
    assert calls == [
        "stm_read",
        "resolve_query_reference",
        "ltm_recall",
        "compose_memory_draft",
        "verify_memory",
        "rewrite_query_for_retrieval",
        "retrieve_phase1",
        "rerank_phase1",
        "compose_with_context",
        "strict_verify",
        "finalize",
        "stm_write",
        "stm_summary",
        "persist_ltm",
    ]


def test_runtime_v2_phase1_fail_goes_degrade(monkeypatch):
    calls = []

    def stm_read_node(state):
        calls.append("stm_read")
        state["short_term_memory"] = "stm"
        state["recent_messages"] = []
        return state

    def resolve_query_reference_node(state):
        calls.append("resolve_query_reference")
        state["resolved_query"] = state.get("query", "")
        return state

    def ltm_recall_node(state):
        calls.append("ltm_recall")
        state["long_term_memory"] = "ltm"
        return state

    def compose_memory_draft_node(state):
        calls.append("compose_memory_draft")
        state["draft_answer"] = "weak-memory"
        state["draft_confidence"] = 0.2
        state["used_memory_chunks"] = 0
        return state

    def verify_memory_node(state):
        calls.append("verify_memory")
        state["memory_verdict"] = "NEED_RETRIEVE"
        state["memory_reason"] = "memory not enough"
        state["memory_risk_level"] = "HIGH"
        return state

    def rewrite_query_for_retrieval_node(state):
        calls.append("rewrite_query_for_retrieval")
        state["retrieval_query"] = state.get("resolved_query", state.get("query", ""))
        return state

    def retrieve_phase1_node(state):
        calls.append("retrieve_phase1")
        state["phase1_candidates"] = [{"id": "d1"}]
        return state

    def rerank_phase1_node(state):
        calls.append("rerank_phase1")
        state["phase1_reranked"] = [{"id": "d1"}]
        state["context_pool"] = [{"id": "d1"}]
        return state

    def compose_with_context_node(state):
        calls.append("compose_with_context")
        state["response"] = "bad-answer"
        state["response_revision"] = state.get("response_revision", 0) + 1
        return state

    def strict_verify_node(state):
        calls.append("strict_verify")
        state["strict_verdict"] = "FAIL"
        state["failure_type"] = "UNKNOWN_TYPE"
        state["strict_reason"] = "not grounded"
        state["verified_revision"] = state.get("response_revision", 0)
        return state

    def degrade_or_abstain_node(state):
        calls.append("degrade_or_abstain")
        state["run_status"] = "degraded"
        state["response"] = "不确定"
        state["citations"] = []
        return state

    fake_registry = {
        "stm_read": stm_read_node,
        "resolve_query_reference": resolve_query_reference_node,
        "ltm_recall": ltm_recall_node,
        "compose_memory_draft": compose_memory_draft_node,
        "verify_memory": verify_memory_node,
        "rewrite_query_for_retrieval": rewrite_query_for_retrieval_node,
        "retrieve_phase1": retrieve_phase1_node,
        "rerank_phase1": rerank_phase1_node,
        "compose_with_context": compose_with_context_node,
        "strict_verify": strict_verify_node,
        "degrade_or_abstain": degrade_or_abstain_node,
        **_base_tail_nodes(calls),
    }

    monkeypatch.setattr(executor_mod, "NODE_REGISTRY", fake_registry)
    monkeypatch.setattr(AppConfig, "RUNTIME_V2_ENABLED", True)

    out = AgentExecutor().run(session_id="v2-s3", query="q")

    assert out["run_status"] == "degraded"
    assert out["runtime_stage"] == "FINALIZE"
    assert out["transition_count"] == 3
    assert calls == [
        "stm_read",
        "resolve_query_reference",
        "ltm_recall",
        "compose_memory_draft",
        "verify_memory",
        "rewrite_query_for_retrieval",
        "retrieve_phase1",
        "rerank_phase1",
        "compose_with_context",
        "strict_verify",
        "degrade_or_abstain",
        "finalize",
        "stm_write",
        "stm_summary",
        "persist_ltm",
    ]
