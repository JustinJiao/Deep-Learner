from core.llm_call import run_prompt
from llm.prompts.base import PromptContract


class _DummyJsonPrompt(PromptContract):
    READS = ["query"]
    WRITES = ["a", "b"]
    SYSTEM = "system"

    def build_user_prompt(self, state):
        return f"query={state.get('query', '')}"


def test_run_prompt_repairs_truncated_json(monkeypatch):
    class _FakeLLM:
        def chat_completion_with_meta(self, prompt, system_prompt, task):
            return {
                "content": '{"a": "ok", "b": 1',
                "provider": "fake",
                "model": "fake-model",
                "latency_ms": 12.0,
            }

    monkeypatch.setattr("core.llm_call.ResourceFactory.get_llm_service", lambda: _FakeLLM())
    state = {"query": "hello", "llm_trace": []}

    out = run_prompt(_DummyJsonPrompt, state)

    assert out == {"a": "ok", "b": 1}
    assert state["llm_trace"][-1]["parse_mode"] == "json_repaired"

