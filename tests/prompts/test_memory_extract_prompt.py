from llm.prompts.ltm_fact_extract import LTMFactExtractPrompt
from ._helpers import (
    assert_validate_reads_ok,
    assert_validate_reads_fail,
    assert_build_prompt_ok,
)


def test_ltm_fact_extract_reads_ok():
    state = {
        "query": "我住在北京，喜欢蓝色",
        "response": "我记住了：你住在北京，喜欢蓝色。",
    }
    assert_validate_reads_ok(LTMFactExtractPrompt, state)


def test_ltm_fact_extract_reads_fail():
    state = {}
    assert_validate_reads_fail(LTMFactExtractPrompt, state)


def test_ltm_fact_extract_build_prompt():
    state = {
        "query": "我用的是 64GB 内存",
        "response": "你可以调高 executor-memory。",
    }
    assert_build_prompt_ok(LTMFactExtractPrompt, state)
