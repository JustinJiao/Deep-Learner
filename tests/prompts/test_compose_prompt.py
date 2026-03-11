from llm .prompts .compose import ComposePrompt 
from ._helpers import (
assert_validate_reads_ok ,
assert_validate_reads_fail ,
assert_build_prompt_ok ,
)

def test_compose_reads_ok ():
    state ={
    "query":"How to adjust Spark memory",
    "short_term_memory":"",
    "recent_messages":[],
    "context_pool":[],
    "long_term_memory":"",
    "is_direct_path":False ,
    "rewritten_query":"",
    "repair_hint":"",
    }
    assert_validate_reads_ok (ComposePrompt ,state )


def test_compose_reads_fail ():
    state ={
    "query":"How to adjust Spark memory",
    "context_pool":[],
    }
    assert_validate_reads_fail (ComposePrompt ,state )


def test_compose_build_prompt_empty_context ():
    state ={
    "query":"How to adjust Spark memory",
    "short_term_memory":"",
    "recent_messages":[],
    "context_pool":[],
    "long_term_memory":"",
    "is_direct_path":False ,
    "rewritten_query":"",
    "repair_hint":"",
    }
    assert_build_prompt_ok (ComposePrompt ,state )


def test_compose_build_prompt_with_docs ():
    state ={
    "query":"How to adjust Spark memory",
    "short_term_memory":"",
    "recent_messages":[],
    "context_pool":[
    {"id":"doc1","content":"Spark uses the JVM memory model"},
    {"id":"doc2","content":"executor memory parameter description"},
    ],
    "long_term_memory":"User cluster memory 64GB",
    "is_direct_path":False ,
    "rewritten_query":"spark executor memory tuning",
    "repair_hint":"",
    }
    assert_build_prompt_ok (ComposePrompt ,state )
