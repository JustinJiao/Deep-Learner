from llm.prompts.rewrite import RewritePrompt
from ._helpers import (
    assert_validate_reads_ok,
    assert_validate_reads_fail,
    assert_build_prompt_ok,
)

def test_rewrite_reads_ok():
    state = {
        "query": "Spark 内存怎么调"
    }
    assert_validate_reads_ok(RewritePrompt, state)


def test_rewrite_build_prompt():
    state = {
        "query": "Spark 内存怎么调",
        "long_term_memory": "用户使用 64GB 内存"
    }
    assert_build_prompt_ok(RewritePrompt, state)
