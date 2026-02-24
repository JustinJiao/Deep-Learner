from config.settings import AppConfig
from nodes.ltm_recall import ltm_recall_node
import nodes.ltm_recall as ltm_recall_mod


def test_ltm_recall_skip_when_top_k_zero(monkeypatch):
    class NeverInitLTM:
        def __init__(self):
            raise AssertionError("LTM should not be initialized when top_k <= 0")

    monkeypatch.setattr(AppConfig, "LTM_RECALL_TOP_K", 0)
    monkeypatch.setattr(ltm_recall_mod, "_LTM_INSTANCE", None)
    monkeypatch.setattr(ltm_recall_mod, "LTM", NeverInitLTM)

    state = {
        "query": "q",
        "resolved_query": "resolved-q",
        "steps_log": [],
    }
    out = ltm_recall_node(state)

    assert out["ltm_hits_count"] == 0
    assert out["long_term_memory"] == "无相关长期记忆"
    assert out["steps_log"][-1].node == "ltm_recall"
    assert out["steps_log"][-1].info["memory"]["top_k"] == 0


def test_ltm_recall_respects_config_top_k(monkeypatch):
    calls = {}

    class FakeLTM:
        def recall(self, query, top_k):
            calls["query"] = query
            calls["top_k"] = top_k
            return ["fact-1", "fact-2"]

    monkeypatch.setattr(AppConfig, "LTM_RECALL_TOP_K", 3)
    monkeypatch.setattr(ltm_recall_mod, "_LTM_INSTANCE", None)
    monkeypatch.setattr(ltm_recall_mod, "LTM", FakeLTM)

    state = {
        "query": "q",
        "resolved_query": "resolved-q",
        "steps_log": [],
    }
    out = ltm_recall_node(state)

    assert calls == {"query": "resolved-q", "top_k": 3}
    assert out["ltm_hits_count"] == 2
    assert out["long_term_memory"] == "fact-1 | fact-2"
    assert out["steps_log"][-1].node == "ltm_recall"
    assert out["steps_log"][-1].info["memory"]["top_k"] == 3
