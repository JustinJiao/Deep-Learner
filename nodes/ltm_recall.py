import time 

from config .settings import AppConfig 
from core .state import AgentState ,StepLog 
from memory .ltm import LTM 
from nodes .log_utils import clip_text 

_LTM_INSTANCE :LTM |None =None 


def _get_ltm ()->LTM :
    global _LTM_INSTANCE 
    if _LTM_INSTANCE is None :
        _LTM_INSTANCE =LTM ()
    return _LTM_INSTANCE 


def ltm_recall_node (state :AgentState )->AgentState :
    recall_query =str (state .get ("resolved_query")or state .get ("query","")).strip ()
    top_k =int (AppConfig .LTM_RECALL_TOP_K )

    if top_k <=0 :
        memories :list [str ]=[]
    else :
        memories =_get_ltm ().recall (recall_query ,top_k =top_k )

    state ["long_term_memory"]=" | ".join (memories )if memories else "No associated long-term memory"
    state ["ltm_hits_count"]=len (memories )

    state .setdefault ("steps_log",[]).append (
    StepLog (
    node ="ltm_recall",
    info ={
    "state":{
    "query_preview":clip_text (state .get ("query",""),180 ),
    "recall_query_preview":clip_text (recall_query ,180 ),
    },
    "memory":{
    "top_k":max (top_k ,0 ),
    "hits_count":len (memories ),
    "memories_preview":[clip_text (m ,120 )for m in memories [:4 ]],
    },
    },
    timestamp =time .time (),
    )
    )
    return state 
