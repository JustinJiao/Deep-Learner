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


def _base_v2_prefix_nodes(calls):
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
        state["phase1_candidates"] = [{"id": "p1"}]
        return state

    def rerank_phase1_node(state):
        calls.append("rerank_phase1")
        state["phase1_reranked"] = [{"id": "p1"}]
        state["context_pool"] = [{"id": "p1"}]
        return state

    return {
        "stm_read": stm_read_node,
        "resolve_query_reference": resolve_query_reference_node,
        "ltm_recall": ltm_recall_node,
        "compose_memory_draft": compose_memory_draft_node,
        "verify_memory": verify_memory_node,
        "rewrite_query_for_retrieval": rewrite_query_for_retrieval_node,
        "retrieve_phase1": retrieve_phase1_node,
        "rerank_phase1": rerank_phase1_node,
    }


def test_runtime_v2_phase2_path_then_pass(monkeypatch):
    calls = []
    strict_calls = {"n": 0}

    def compose_with_context_node(state):
        calls.append("compose_with_context")
        state["response"] = f"answer-rev-{state.get('response_revision', 0) + 1}"
        state["response_revision"] = state.get("response_revision", 0) + 1
        return state

    def strict_verify_node(state):
        calls.append("strict_verify")
        strict_calls["n"] += 1
        if strict_calls["n"] == 1:
            state["strict_verdict"] = "FAIL"
            state["failure_type"] = "INSUFFICIENT_EVIDENCE"
            state["strict_reason"] = "phase1 evidence weak"
        else:
            state["strict_verdict"] = "PASS"
            state["strict_reason"] = "phase2 grounded"
        state["verified_revision"] = state.get("response_revision", 0)
        return state

    def retrieve_phase2_node(state):
        calls.append("retrieve_phase2")
        state["phase2_candidates"] = [{"id": "p2"}]
        return state

    def rerank_phase2_node(state):
        calls.append("rerank_phase2")
        state["phase2_reranked"] = [{"id": "p2"}]
        state["context_pool"] = [{"id": "p2"}]
        return state

    fake_registry = {
        **_base_v2_prefix_nodes(calls),
        "compose_with_context": compose_with_context_node,
        "strict_verify": strict_verify_node,
        "retrieve_phase2": retrieve_phase2_node,
        "rerank_phase2": rerank_phase2_node,
        **_base_tail_nodes(calls),
    }

    monkeypatch.setattr(executor_mod, "NODE_REGISTRY", fake_registry)
    monkeypatch.setattr(AppConfig, "RUNTIME_V2_ENABLED", True)

    out = AgentExecutor().run(session_id="v2-p3-s1", query="q")

    assert out["run_status"] == "ok"
    assert out["runtime_stage"] == "FINALIZE"
    assert out["phase2_used"] is True
    assert out["repair_used"] is False
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
        "retrieve_phase2",
        "rerank_phase2",
        "compose_with_context",
        "strict_verify",
        "finalize",
        "stm_write",
        "stm_summary",
        "persist_ltm",
    ]


def test_runtime_v2_phase2_fail_goes_degrade(monkeypatch):
    calls = []
    strict_calls = {"n": 0}

    def compose_with_context_node(state):
        calls.append("compose_with_context")
        state["response"] = f"answer-rev-{state.get('response_revision', 0) + 1}"
        state["response_revision"] = state.get("response_revision", 0) + 1
        return state

    def strict_verify_node(state):
        calls.append("strict_verify")
        strict_calls["n"] += 1
        if strict_calls["n"] == 1:
            state["strict_verdict"] = "FAIL"
            state["failure_type"] = "INSUFFICIENT_EVIDENCE"
            state["strict_reason"] = "phase1 evidence weak"
        else:
            state["strict_verdict"] = "FAIL"
            state["failure_type"] = "LOGICAL_ERROR"
            state["strict_reason"] = "phase2 still bad"
        state["verified_revision"] = state.get("response_revision", 0)
        return state

    def retrieve_phase2_node(state):
        calls.append("retrieve_phase2")
        state["phase2_candidates"] = [{"id": "p2"}]
        return state

    def rerank_phase2_node(state):
        calls.append("rerank_phase2")
        state["phase2_reranked"] = [{"id": "p2"}]
        state["context_pool"] = [{"id": "p2"}]
        return state

    def degrade_or_abstain_node(state):
        calls.append("degrade_or_abstain")
        state["run_status"] = "degraded"
        state["response"] = "不确定"
        state["citations"] = []
        return state

    fake_registry = {
        **_base_v2_prefix_nodes(calls),
        "compose_with_context": compose_with_context_node,
        "strict_verify": strict_verify_node,
        "retrieve_phase2": retrieve_phase2_node,
        "rerank_phase2": rerank_phase2_node,
        "degrade_or_abstain": degrade_or_abstain_node,
        **_base_tail_nodes(calls),
    }

    monkeypatch.setattr(executor_mod, "NODE_REGISTRY", fake_registry)
    monkeypatch.setattr(AppConfig, "RUNTIME_V2_ENABLED", True)

    out = AgentExecutor().run(session_id="v2-p3-s2", query="q")

    assert out["run_status"] == "degraded"
    assert out["runtime_stage"] == "FINALIZE"
    assert out["phase2_used"] is True
    assert out["repair_used"] is False
    assert out["transition_count"] == 4
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
        "retrieve_phase2",
        "rerank_phase2",
        "compose_with_context",
        "strict_verify",
        "degrade_or_abstain",
        "finalize",
        "stm_write",
        "stm_summary",
        "persist_ltm",
    ]


def test_runtime_v2_repair_mode_then_pass(monkeypatch):
    calls = []
    strict_calls = {"n": 0}
    compose_modes = []

    def compose_with_context_node(state):
        calls.append("compose_with_context")
        compose_modes.append(bool(state.get("repair_mode", False)))
        state["response"] = f"answer-rev-{state.get('response_revision', 0) + 1}"
        state["response_revision"] = state.get("response_revision", 0) + 1
        return state

    def strict_verify_node(state):
        calls.append("strict_verify")
        strict_calls["n"] += 1
        if strict_calls["n"] == 1:
            state["strict_verdict"] = "FAIL"
            state["failure_type"] = "CITATION_MISMATCH"
            state["strict_reason"] = "citation mismatch"
        else:
            state["strict_verdict"] = "PASS"
            state["strict_reason"] = "repaired"
        state["verified_revision"] = state.get("response_revision", 0)
        return state

    def set_repair_mode_node(state):
        calls.append("set_repair_mode")
        state["repair_mode"] = True
        state["repair_used"] = True
        state["repair_reason"] = "CITATION_MISMATCH: citation mismatch"
        return state

    fake_registry = {
        **_base_v2_prefix_nodes(calls),
        "compose_with_context": compose_with_context_node,
        "strict_verify": strict_verify_node,
        "set_repair_mode": set_repair_mode_node,
        **_base_tail_nodes(calls),
    }

    monkeypatch.setattr(executor_mod, "NODE_REGISTRY", fake_registry)
    monkeypatch.setattr(AppConfig, "RUNTIME_V2_ENABLED", True)

    out = AgentExecutor().run(session_id="v2-p3-s3", query="q")

    assert out["run_status"] == "ok"
    assert out["runtime_stage"] == "FINALIZE"
    assert out["phase2_used"] is False
    assert out["repair_used"] is True
    assert out["transition_count"] == 3
    assert compose_modes == [False, True]
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
        "set_repair_mode",
        "compose_with_context",
        "strict_verify",
        "finalize",
        "stm_write",
        "stm_summary",
        "persist_ltm",
    ]
