from llm.prompts.base import PromptContractError
from nodes.compose_memory_draft import compose_memory_draft_node


def _base_state():
    return {
        "query": "Which cloud platform belongs to each company?",
        "resolved_query": "Which cloud platform belongs to each company?",
        "short_term_memory": "Amazon has AWS.",
        "recent_messages": [],
        "long_term_memory": "Alphabet has Google Cloud. Microsoft has Azure.",
        "ltm_hits_count": 2,
        "steps_log": [],
        "llm_trace": [],
    }


def test_compose_memory_draft_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def _fake_run_prompt(_prompt_cls, _state):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PromptContractError("invalid json")
        return {
            "draft_answer": "AWS belongs to Amazon.",
            "confidence": 0.78,
            "used_memory_chunks": 2,
        }

    monkeypatch.setattr("nodes.compose_memory_draft.run_prompt", _fake_run_prompt)
    monkeypatch.setattr("nodes.compose_memory_draft.AppConfig.RUNTIME_MEMORY_DRAFT_MAX_RETRIES", 1)

    out = compose_memory_draft_node(_base_state())

    assert calls["n"] == 2
    assert out["draft_answer"] == "AWS belongs to Amazon."
    assert out["draft_confidence"] == 0.78
    assert out["used_memory_chunks"] == 2
    llm_output = out["steps_log"][-1].info["llm_output"]
    assert llm_output["llm_attempts"] == 2
    assert llm_output["used_fallback"] is False


def test_compose_memory_draft_fallback_on_contract_error(monkeypatch):
    def _fake_run_prompt(_prompt_cls, _state):
        raise PromptContractError("invalid json")

    monkeypatch.setattr("nodes.compose_memory_draft.run_prompt", _fake_run_prompt)
    monkeypatch.setattr("nodes.compose_memory_draft.AppConfig.RUNTIME_MEMORY_DRAFT_MAX_RETRIES", 1)

    out = compose_memory_draft_node(_base_state())

    assert out["draft_answer"] == "记忆证据不足，需检索外部文档。"
    assert out["draft_confidence"] == 0.0
    assert out["used_memory_chunks"] == 0
    llm_output = out["steps_log"][-1].info["llm_output"]
    assert llm_output["llm_attempts"] == 2
    assert llm_output["used_fallback"] is True
    assert "PromptContractError" in llm_output["llm_error_preview"]

