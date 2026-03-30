from __future__ import annotations 

from dataclasses import dataclass ,field 
from typing import Any ,Callable 

from nodes .persist_ltm import persist_ltm_node 
from nodes .finalize import finalize_node 
from nodes .stm_read import stm_read_node 
from nodes .stm_write import stm_write_node 
from nodes .stm_summary import stm_summary_node 
from nodes .ltm_recall import ltm_recall_node 
from nodes .resolve_query_reference import resolve_query_reference_node 
from nodes .rewrite_query_for_retrieval import rewrite_query_for_retrieval_node 
from nodes .compose_memory_draft import compose_memory_draft_node 
from nodes .verify_memory import verify_memory_node 
from nodes .retrieve_phase1 import retrieve_phase1_node 
from nodes .rerank_phase1 import rerank_phase1_node 
from nodes .compose_with_context import compose_with_context_node 
from nodes .strict_verify import strict_verify_node 
from nodes .degrade_or_abstain import degrade_or_abstain_node 
from nodes .retrieve_phase2 import retrieve_phase2_node 
from nodes .rerank_phase2 import rerank_phase2_node 
from nodes .set_repair_mode import set_repair_mode_node 
from nodes .extract_route_facts import extract_route_facts_node 


@dataclass (frozen =True )
class NodeContract :
    name :str 
    reads :set [str ]=field (default_factory =set )
    writes :set [str ]=field (default_factory =set )
    llm_node :bool =False 
    # PR1 accesses in non-strict mode by default to avoid affecting existing links; subsequent PRs can gradually turn on strict=True
    strict :bool =False 


NODE_REGISTRY :dict [str ,Callable [...,Any ]]={
"stm_read":stm_read_node ,
"finalize":finalize_node ,
"stm_write":stm_write_node ,
"stm_summary":stm_summary_node ,
"persist_ltm":persist_ltm_node ,
# Runtime V2 nodes
"resolve_query_reference":resolve_query_reference_node ,
"rewrite_query_for_retrieval":rewrite_query_for_retrieval_node ,
"ltm_recall":ltm_recall_node ,
"compose_memory_draft":compose_memory_draft_node ,
"verify_memory":verify_memory_node ,
"retrieve_phase1":retrieve_phase1_node ,
"rerank_phase1":rerank_phase1_node ,
"extract_route_facts":extract_route_facts_node ,
"compose_with_context":compose_with_context_node ,
"strict_verify":strict_verify_node ,
"degrade_or_abstain":degrade_or_abstain_node ,
"retrieve_phase2":retrieve_phase2_node ,
"rerank_phase2":rerank_phase2_node ,
"set_repair_mode":set_repair_mode_node ,
}


NODE_CONTRACTS :dict [str ,NodeContract ]={
"stm_read":NodeContract (
name ="stm_read",
reads ={"session_id"},
writes ={"short_term_memory","recent_messages","_stm_to_compress","steps_log"},
),
"finalize":NodeContract (
name ="finalize",
reads ={"response"},
writes ={"response"},
),
"stm_write":NodeContract (
name ="stm_write",
reads ={"response","session_id"},
writes ={"steps_log"},
),
"stm_summary":NodeContract (
name ="stm_summary",
reads ={"session_id"},
writes ={"steps_log"},
),
"persist_ltm":NodeContract (
name ="persist_ltm",
reads ={"query","response"},
writes ={"steps_log"},
),
# Runtime V2 contracts
"resolve_query_reference":NodeContract (
name ="resolve_query_reference",
reads ={"query","short_term_memory","recent_messages","long_term_memory"},
writes ={"resolved_query","steps_log"},
llm_node =True ,
),
"rewrite_query_for_retrieval":NodeContract (
name ="rewrite_query_for_retrieval",
reads ={"query","resolved_query","short_term_memory","recent_messages","long_term_memory"},
writes ={"retrieval_query","retrieval_queries","steps_log"},
llm_node =True ,
),
"ltm_recall":NodeContract (
name ="ltm_recall",
reads ={"query","resolved_query"},
writes ={"long_term_memory","ltm_hits_count","steps_log"},
),
"compose_memory_draft":NodeContract (
name ="compose_memory_draft",
reads ={"query","short_term_memory","recent_messages","long_term_memory","ltm_hits_count"},
writes ={"draft_answer","draft_confidence","used_memory_chunks","steps_log"},
llm_node =True ,
),
"verify_memory":NodeContract (
name ="verify_memory",
reads ={
"query",
"resolved_query",
"draft_answer",
"draft_confidence",
"used_memory_chunks",
"ltm_hits_count",
"short_term_memory",
"recent_messages",
"long_term_memory",
},
writes ={"memory_score","memory_verdict","memory_reason","memory_risk_level","steps_log"},
llm_node =True ,
),
"retrieve_phase1":NodeContract (
name ="retrieve_phase1",
reads ={"query","resolved_query","retrieval_query","retrieval_queries"},
writes ={"phase1_candidates","phase1_query_routes","steps_log"},
),
"rerank_phase1":NodeContract (
name ="rerank_phase1",
reads ={"query","retrieval_query","retrieval_queries","phase1_candidates","phase1_query_routes"},
writes ={
"phase1_candidates",
"phase1_query_routes",
"phase1_reranked",
"context_pool",
"context_source",
"steps_log",
},
),
"extract_route_facts":NodeContract (
name ="extract_route_facts",
reads ={
"query",
"resolved_query",
"retrieval_queries",
"phase1_query_routes",
"phase2_query_routes",
"context_pool",
},
writes ={"route_facts","route_fact_coverage","steps_log"},
),
"compose_with_context":NodeContract (
name ="compose_with_context",
reads ={
"query",
"short_term_memory",
"recent_messages",
"long_term_memory",
"context_pool",
"phase1_query_routes",
"phase2_query_routes",
"repair_mode",
"failure_type",
"strict_reason",
"previous_response",
"evidence_table",
"route_facts",
"route_fact_coverage",
},
writes ={"response","citations","response_revision","previous_response","evidence_table","steps_log"},
llm_node =True ,
),
"strict_verify":NodeContract (
name ="strict_verify",
reads ={"query","resolved_query","response","citations","context_pool","response_revision","evidence_table"},
writes ={
"strict_score",
"strict_total_score",
"strict_action",
"strict_metrics",
"strict_confidence",
"strict_verdict",
"repair_trigger",
"failure_type",
"strict_reason",
"verified_revision",
"steps_log",
},
llm_node =True ,
),
"degrade_or_abstain":NodeContract (
name ="degrade_or_abstain",
reads ={
"strict_reason",
"failure_type",
"repair_trigger",
"response",
"previous_response",
"citations",
"route_facts",
"route_fact_coverage",
},
writes ={"response","citations","run_status","strict_status","steps_log"},
),
"retrieve_phase2":NodeContract (
name ="retrieve_phase2",
reads ={"query","resolved_query","retrieval_query","retrieval_queries"},
writes ={"phase2_candidates","phase2_query_routes","phase1_query_routes","steps_log"},
),
"rerank_phase2":NodeContract (
name ="rerank_phase2",
reads ={
"query",
"retrieval_query",
"retrieval_queries",
"phase2_candidates",
"phase2_query_routes",
"phase1_query_routes",
},
writes ={
"phase2_candidates",
"phase2_query_routes",
"phase1_query_routes",
"phase2_reranked",
"context_pool",
"context_source",
"steps_log",
},
),
"set_repair_mode":NodeContract (
name ="set_repair_mode",
reads ={"strict_verdict","failure_type","strict_reason","repair_trigger","repair_used"},
writes ={"repair_mode","repair_used","repair_reason","steps_log"},
),
}


def _changed_keys (before_state :dict [str ,Any ],after_state :dict [str ,Any ])->set [str ]:
    changed :set [str ]=set ()
    keys =set (before_state .keys ())|set (after_state .keys ())
    for key in keys :
        before =before_state .get (key ,None )
        after =after_state .get (key ,None )
        if key not in before_state or key not in after_state or before !=after :
            changed .add (key )
    return changed 


def validate_node_contract (
node_name :str ,
before_state :dict [str ,Any ],
after_state :dict [str ,Any ],
contracts :dict [str ,NodeContract ]|None =None ,
)->dict [str ,Any ]:
    contract_map =contracts or NODE_CONTRACTS 
    contract =contract_map .get (node_name )
    if contract is None :
        return {
        "node":node_name ,
        "enforced":False ,
        "valid":True ,
        "missing_reads":[],
        "unexpected_writes":[],
        "changed_keys":[],
        }

    missing_reads =sorted (k for k in contract .reads if k not in before_state )
    changed_keys =sorted (_changed_keys (before_state ,after_state ))
    unexpected_writes =sorted (k for k in changed_keys if k not in contract .writes )

    # When strict=False, only observations are made and execution is not intercepted.
    valid =(not missing_reads )and (not unexpected_writes or not contract .strict )

    return {
    "node":node_name ,
    "enforced":bool (contract .strict ),
    "valid":valid ,
    "missing_reads":missing_reads ,
    "unexpected_writes":unexpected_writes ,
    "changed_keys":changed_keys ,
    }
