import time 
import re 

from config .settings import AppConfig 
from core .llm_call import run_prompt 
from core .state import AgentState ,StepLog 
from llm .prompts .base import PromptContractError 
from llm .prompts .rewrite_retrieval_query import RewriteRetrievalQueryPrompt 
from nodes .log_utils import clip_text ,preview_messages 


def _base_query (state :AgentState )->str :
    return str (state .get ("resolved_query")or state .get ("query")or "").strip ()


def _normalize_query_list (values :object )->list [str ]:
    if not isinstance (values ,list ):
        return []
    deduped :list [str ]=[]
    seen :set [str ]=set ()
    max_queries =max (1 ,int (AppConfig .RETRIEVAL_MULTI_QUERY_MAX_QUERIES ))
    for item in values :
        text =" ".join (str (item or "").split ()).strip ()
        lowered =text .lower ()
        if not text or lowered in seen :
            continue 
        seen .add (lowered )
        deduped .append (text )
        if len (deduped )>=max_queries :
            break 
    return deduped 


def _align_with_original_scope (queries :list [str ],original_query :str )->list [str ]:
    original_lower =str (original_query or "").lower ()
    aligned :list [str ]=[]
    for q in queries :
        text =str (q or "").strip ()
        lowered =text .lower ()
        if "azure"in lowered and "azure"not in original_lower :
            text =re .sub (r"\bazure\b","Microsoft Cloud",text ,flags =re .IGNORECASE )
            text =re .sub (
            r"\bmicrosoft\s+microsoft\s+cloud\b",
            "Microsoft Cloud",
            text ,
            flags =re .IGNORECASE ,
            )
        aligned .append (" ".join (text .split ()))
    return aligned 


def rewrite_query_for_retrieval_node (state :AgentState )->AgentState :
    base_query =_base_query (state )
    llm_error =""
    used_fallback =False 
    out :dict ={}

    try :
        out =run_prompt (RewriteRetrievalQueryPrompt ,state )
    except PromptContractError as e :
    # When the query rewrite fails, the main process is not blocked and falls back directly to base_query.
        llm_error =f"{type(e).__name__}: {e}"
        used_fallback =True 

    retrieval_queries =_normalize_query_list (out .get ("retrieval_queries"))

    # Backward compatibility for legacy prompt output field.
    legacy_single_query =" ".join (str (out .get ("retrieval_query","")).split ()).strip ()
    if not retrieval_queries and legacy_single_query :
        retrieval_queries =[legacy_single_query ]

    if not retrieval_queries :
        retrieval_queries =[base_query ]if base_query else []
        used_fallback =True 

    retrieval_queries =_align_with_original_scope (
    queries =retrieval_queries ,
    original_query =str (state .get ("query","")or ""),
    )
    retrieval_query =retrieval_queries [0 ]if retrieval_queries else base_query 

    state ["retrieval_query"]=retrieval_query 
    state ["retrieval_queries"]=retrieval_queries 

    state .setdefault ("steps_log",[]).append (
    StepLog (
    node ="rewrite_query_for_retrieval",
    info ={
    "state":{
    "query_preview":clip_text (str (state .get ("query","")),180 ),
    "resolved_query_preview":clip_text (str (state .get ("resolved_query","")),180 ),
    },
    "llm_input":{
    "long_term_memory_preview":clip_text (state .get ("long_term_memory",""),160 ),
    "recent_messages_preview":preview_messages (state .get ("recent_messages",[])),
    },
    "llm_output":{
    "base_query_preview":clip_text (base_query ,180 ),
    "retrieval_query_preview":clip_text (retrieval_query ,180 ),
    "retrieval_queries":[clip_text (q ,180 )for q in retrieval_queries ],
    "retrieval_query_count":len (retrieval_queries ),
    "used_fallback":used_fallback ,
    "legacy_single_query_preview":clip_text (legacy_single_query ,160 ),
    "llm_error_preview":clip_text (llm_error ,220 ),
    },
    },
    timestamp =time .time (),
    )
    )
    return state 
