from llm .prompts .intent import IntentPrompt 
from ._helpers import (
assert_validate_reads_ok ,
assert_validate_reads_fail ,
assert_build_prompt_ok ,
)

def test_intent_reads_ok ():
    state ={
    "query":"Hello, I am making a RAG system",
    "short_term_memory":"",
    "recent_messages":[],
    }
    assert_validate_reads_ok (IntentPrompt ,state )


def test_intent_reads_fail ():
    state ={}
    assert_validate_reads_fail (IntentPrompt ,state )


def test_intent_build_prompt ():
    state ={
    "query":"Please explain what is vector retrieval",
    "short_term_memory":"",
    "recent_messages":[],
    }
    assert_build_prompt_ok (IntentPrompt ,state )
