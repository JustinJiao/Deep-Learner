from llm .prompts .rewrite import RewritePrompt 
from ._helpers import (
assert_validate_reads_ok ,
assert_validate_reads_fail ,
assert_build_prompt_ok ,
)

def test_rewrite_reads_ok ():
    state ={
    "query":"How to adjust Spark memory",
    "short_term_memory":"",
    "recent_messages":[],
    "long_term_memory":"",
    }
    assert_validate_reads_ok (RewritePrompt ,state )


def test_rewrite_build_prompt ():
    state ={
    "query":"How to adjust Spark memory",
    "short_term_memory":"",
    "recent_messages":[],
    "long_term_memory":"User uses 64GB RAM",
    }
    assert_build_prompt_ok (RewritePrompt ,state )
