# core/state.py

import time 
from typing import List ,Dict ,Any ,Literal ,Annotated 
from typing_extensions import TypedDict 
from pydantic import BaseModel ,Field 
from operator import add 
from core .plan import ExecutionPlan 


class StepLog (BaseModel ):
    node :str 
    info :Any 
    timestamp :float =Field (default_factory =time .time )


class Intent (TypedDict ):
    type :Literal ["chat","research"]
    confidence :float 


RuntimeStage =Literal [
"MEMORY",
"PHASE1",
"PHASE2",
"REPAIR",
"DEGRADE",
"FINALIZE",
]

StrictFailureType =Literal [
"INSUFFICIENT_EVIDENCE",
"LOGICAL_ERROR",
"CITATION_MISMATCH",
"FORMAT_ERROR",
]


class STMState (TypedDict ):
    summary :List [str ]
    messages :List [Dict [str ,Any ]]
    recent_messages :List [Dict [str ,Any ]]
    compressed_until :int 


class AgentState (TypedDict ,total =False ):
# enter
    query :str 
    session_id :str 

    # Running status
    run_status :Literal ["running","ok","degraded","error"]

    # Memory projection field (for LLM)
    short_term_memory :str 
    recent_messages :List [Dict [str ,Any ]]
    long_term_memory :str 

    # control flow
    intent :Intent 
    is_direct_path :bool 
    plan :ExecutionPlan 
    loop_count :int 

    # RAG
    rewritten_query :str 
    resolved_query :str 
    retrieval_query :str 
    retrieval_queries :List [str ]
    context_pool :List [Dict [str ,Any ]]
    evidence_table :List [Dict [str ,Any ]]

    # output
    response :str 
    citations :List [Dict [str ,Any ]]
    previous_response :str 

    # V2 Runtime control field (PR1: introduce the contract first, and then implement the business logic with PR)
    runtime_stage :RuntimeStage 
    transition_count :int 
    phase2_used :bool 
    repair_used :bool 
    repair_mode :bool 
    response_revision :int 
    verified_revision :int 
    verify_score :float 
    strict_score :float 
    strict_total_score :float 
    memory_score :float 
    strict_verdict :Literal ["PASS","FAIL"]
    strict_action :Literal ["PASS","REPAIR"]
    strict_status :Literal ["PASS","REPAIRED","FAILED"]
    strict_metrics :Dict [str ,Any ]
    strict_confidence :float 
    repair_trigger :str 
    failure_type :StrictFailureType 
    strict_reason :str 
    ltm_hits_count :int 
    draft_answer :str 
    draft_confidence :float 
    used_memory_chunks :int 
    memory_verdict :Literal ["SUFFICIENT","NEED_RETRIEVE"]
    memory_reason :str 
    memory_risk_level :Literal ["LOW","MEDIUM","HIGH"]
    phase1_candidates :List [Dict [str ,Any ]]
    phase1_query_routes :List [Dict [str ,Any ]]
    phase2_query_routes :List [Dict [str ,Any ]]
    route_facts :List [Dict [str ,Any ]]
    route_fact_coverage :Dict [str ,Any ]
    phase1_reranked :List [Dict [str ,Any ]]
    phase2_candidates :List [Dict [str ,Any ]]
    phase2_reranked :List [Dict [str ,Any ]]
    context_source :Literal ["phase1","phase2"]
    repair_reason :str 

    # Verify / Repair
    is_hallucination :bool 
    critique :Dict [str ,Any ]# VerifyPrompt output (contains error_type/next_step/critique)
    repair_hint :str 

    # Debug
    steps_log :Annotated [List [StepLog ],add ]

    # internal fields
    _stm_to_compress :List [Dict [str ,Any ]]


def build_initial_state (session_id :str ,query :str )->AgentState :
    return {
    "session_id":session_id ,
    "query":query ,
    "steps_log":[],
    "loop_count":0 ,
    "run_status":"running",
    # V2 runtime defaults
    "runtime_stage":"MEMORY",
    "transition_count":0 ,
    "phase2_used":False ,
    "repair_used":False ,
    "repair_mode":False ,
    "response_revision":0 ,
    "verified_revision":0 ,
    "verify_score":0.0 ,
    "strict_score":0.0 ,
    "strict_total_score":0.0 ,
    "memory_score":0.0 ,
    "strict_action":"PASS",
    "strict_status":"PASS",
    "strict_metrics":{},
    "strict_confidence":0.0 ,
    "repair_trigger":"",
    "citations":[],
    "evidence_table":[],
    "repair_reason":"",
    "phase1_query_routes":[],
    "phase2_query_routes":[],
    "route_facts":[],
    "route_fact_coverage":{},
    }
