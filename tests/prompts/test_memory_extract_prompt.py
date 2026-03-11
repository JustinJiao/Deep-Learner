from llm .prompts .ltm_fact_extract import LTMFactExtractPrompt 
from ._helpers import (
assert_validate_reads_ok ,
assert_validate_reads_fail ,
assert_build_prompt_ok ,
)


def test_ltm_fact_extract_reads_ok ():
    state ={
    "query":"I live in Beijing and like blue",
    "response":"I remember: you live in Beijing and like blue.",
    }
    assert_validate_reads_ok (LTMFactExtractPrompt ,state )


def test_ltm_fact_extract_reads_fail ():
    state ={}
    assert_validate_reads_fail (LTMFactExtractPrompt ,state )


def test_ltm_fact_extract_build_prompt ():
    state ={
    "query":"I am using 64GB RAM",
    "response":"You can increase executor-memory.",
    }
    assert_build_prompt_ok (LTMFactExtractPrompt ,state )
