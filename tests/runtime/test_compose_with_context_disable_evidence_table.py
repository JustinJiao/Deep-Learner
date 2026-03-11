from nodes.compose_with_context import compose_with_context_node


def test_compose_with_context_skips_evidence_table_when_disabled(monkeypatch):
    def _fake_run_prompt(prompt_cls, _state):
        name = getattr(prompt_cls, "__name__", "")
        if name == "ComposeEvidenceTablePrompt":
            raise AssertionError("ComposeEvidenceTablePrompt should not be called when disabled")
        if name == "ComposeWithContextPrompt":
            return {
                "response": "AWS belongs to Amazon.",
                "citations": [
                    {
                        "id": "doc-1",
                        "title": "Amazon 10K 2024.pdf",
                        "score": 1.0,
                        "quote": "AWS ... 107,556",
                    }
                ],
            }
        raise AssertionError(f"unexpected prompt call: {name}")

    monkeypatch.setattr("nodes.compose_with_context.run_prompt", _fake_run_prompt)
    monkeypatch.setattr("nodes.compose_with_context.AppConfig.RUNTIME_ENABLE_EVIDENCE_TABLE", False)
    monkeypatch.setattr("nodes.compose_with_context.AppConfig.RUNTIME_FORCE_ANSWER_ON_EVIDENCE", False)

    state = {
        "query": "Which cloud platform belongs to Amazon?",
        "resolved_query": "Which cloud platform belongs to Amazon?",
        "retrieval_query": "amazon cloud platform",
        "retrieval_queries": ["amazon cloud platform"],
        "short_term_memory": "",
        "recent_messages": [],
        "long_term_memory": "",
        "context_pool": [
            {
                "id": "doc-1",
                "title": "Amazon 10K 2024.pdf",
                "source": "Amazon 10K 2024.pdf",
                "score": 0.9,
                "content": "AWS ... 107,556",
                "metadata": {"source": "Amazon 10K 2024.pdf"},
            }
        ],
        "phase1_query_routes": [],
        "repair_mode": False,
        "failure_type": "",
        "strict_reason": "",
        "previous_response": "",
        "evidence_table": [{"company": "Alphabet", "value": "bad stale row"}],
        "steps_log": [],
    }

    out = compose_with_context_node(state)

    assert out["evidence_table"] == []
    assert out["response"] == "AWS belongs to Amazon."
    assert out["citations"]

    log = out["steps_log"][-1]
    assert log.node == "compose_with_context"
    assert log.info["state"]["evidence_table_count"] == 0
    assert (
        log.info["state"]["evidence_table_error_preview"]
        == "disabled_by_config:RUNTIME_ENABLE_EVIDENCE_TABLE=false"
    )

