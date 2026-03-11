from llm .prompts .verify import VerifyPrompt 
from ._helpers import (
assert_validate_reads_ok ,
assert_validate_reads_fail ,
assert_build_prompt_ok ,
)

def test_verify_reads_ok ():
    state ={
    "response":"Spark can be set via --executor-memory",
    "context_pool":[],
    }
    assert_validate_reads_ok (VerifyPrompt ,state )


def test_verify_build_prompt ():
    state ={
    "response":"Spark can be set via --executor-memory",
    "context_pool":[
    {"id":"doc1","content":"executor-memory parameter description"}
    ],
    "long_term_memory":"",
    }
    assert_build_prompt_ok (VerifyPrompt ,state )
