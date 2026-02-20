from nodes.verify import verify_node


def test_verify_node_clears_error_fields_on_pass(monkeypatch):
    def fake_run_prompt(prompt_cls, state):
        return {
            "is_hallucination": False,
            "error_type": "retrieval_insufficient",
            "next_step": "compose",
            "critique": "looks fine",
        }

    monkeypatch.setattr("nodes.verify.run_prompt", fake_run_prompt)

    state = {
        "response": "ok",
        "context_pool": [{"id": "d1", "content": "x", "score": 1.0}],
        "repair_hint": "old hint",
        "steps_log": [],
    }

    out = verify_node(state)

    assert out["is_hallucination"] is False
    assert out["critique"]["error_type"] == ""
    assert out["critique"]["next_step"] == ""
    assert "repair_hint" not in out
    assert out["steps_log"][-1].info["llm_output"]["verdict"] == "PASS"
    assert out["steps_log"][-1].info["llm_output"]["error_type"] == ""
    assert out["steps_log"][-1].info["llm_output"]["next_step"] == ""


def test_verify_node_normalizes_invalid_next_step_on_fail(monkeypatch):
    def fake_run_prompt(prompt_cls, state):
        return {
            "is_hallucination": True,
            "error_type": "retrieval_insufficient",
            "next_step": "invalid-step",
            "critique": "not enough evidence",
        }

    monkeypatch.setattr("nodes.verify.run_prompt", fake_run_prompt)

    state = {
        "response": "ok",
        "context_pool": [{"id": "d1", "content": "x", "score": 1.0}],
        "steps_log": [],
    }

    out = verify_node(state)

    assert out["is_hallucination"] is True
    assert out["critique"]["error_type"] == "retrieval_insufficient"
    assert out["critique"]["next_step"] == "compose"
    assert out["steps_log"][-1].info["llm_output"]["verdict"] == "FAIL"
    assert out["steps_log"][-1].info["llm_output"]["error_type"] == "retrieval_insufficient"
    assert out["steps_log"][-1].info["llm_output"]["next_step"] == "compose"


def test_verify_node_supports_type_reason_alias_fields(monkeypatch):
    def fake_run_prompt(prompt_cls, state):
        return {
            "is_hallucination": True,
            "type": "query_misaligned",
            "reason": "query misses key term",
        }

    monkeypatch.setattr("nodes.verify.run_prompt", fake_run_prompt)

    state = {
        "response": "ok",
        "context_pool": [{"id": "d1", "content": "x", "score": 1.0}],
        "steps_log": [],
    }

    out = verify_node(state)

    assert out["is_hallucination"] is True
    assert out["critique"]["error_type"] == "query_misaligned"
    assert out["critique"]["next_step"] == "compose"
    assert out["critique"]["critique"] == "query misses key term"
    assert out["steps_log"][-1].info["llm_output"]["verdict"] == "FAIL"
    assert out["steps_log"][-1].info["llm_output"]["error_type"] == "query_misaligned"
