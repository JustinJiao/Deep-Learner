from llm.prompts.intent import IntentPrompt
from ._helpers import (
    assert_validate_reads_ok,
    assert_validate_reads_fail,
    assert_build_prompt_ok,
)

def test_intent_reads_ok():
    state = {
        "query": "你好，我在做一个 RAG 系统",
        "short_term_memory": "",
        "recent_messages": [],
    }
    assert_validate_reads_ok(IntentPrompt, state)


def test_intent_reads_fail():
    state = {}
    assert_validate_reads_fail(IntentPrompt, state)


def test_intent_build_prompt():
    state = {
        "query": "请解释什么是向量检索",
        "short_term_memory": "",
        "recent_messages": [],
    }
    assert_build_prompt_ok(IntentPrompt, state)
