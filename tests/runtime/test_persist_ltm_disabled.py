from nodes.persist_ltm import persist_ltm_node


def test_persist_ltm_skips_when_disabled(monkeypatch):
    def _unexpected_run_prompt(*_args, **_kwargs):
        raise AssertionError("run_prompt should not be called when LTM writing is disabled")

    monkeypatch.setattr("nodes.persist_ltm.run_prompt", _unexpected_run_prompt)
    monkeypatch.setattr("nodes.persist_ltm.AppConfig.LTM_WRITE_ENABLED", False)

    state = {
        "query": "q",
        "response": "answer",
        "steps_log": [],
    }
    out = persist_ltm_node(state)

    assert out["steps_log"]
    last = out["steps_log"][-1]
    assert last.node == "persist_ltm"
    assert last.info["state"]["persist_skipped"] is True
    assert last.info["state"]["reason"] == "LTM_WRITE_ENABLED=false"

