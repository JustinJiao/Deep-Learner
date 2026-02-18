from llm.prompts.memory_extract import MemoryExtractPrompt
from ._helpers import (
    assert_validate_reads_ok,
    assert_validate_reads_fail,
    assert_build_prompt_ok,
)

def test_memory_extract_reads_ok():
    state = {
        "messages": [
            ("user", "我用的是 64GB 内存"),
            ("assistant", "那可以调高 executor-memory"),
        ]
    }
    assert_validate_reads_ok(MemoryExtractPrompt, state)


def test_memory_extract_reads_fail():
    state = {}
    assert_validate_reads_fail(MemoryExtractPrompt, state)


def test_memory_extract_build_prompt():
    state = {
        "messages": [
            ("user", "我用的是 64GB 内存"),
            ("assistant", "那可以调高 executor-memory"),
        ]
    }
    assert_build_prompt_ok(MemoryExtractPrompt, state)
