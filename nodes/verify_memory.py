import time 
from typing import Any 

from config .settings import AppConfig 
from core .llm_call import run_prompt 
from core .state import AgentState ,StepLog 
from llm .prompts .verify_memory import VerifyMemoryPrompt 
from nodes .log_utils import clip_text 
from nodes .strict_verify import strict_verify_node 

_UNCERTAIN_MARKERS =(
"uncertain",
"Unable to determine",
"Insufficient evidence",
"Need to search",
"uncertain",
"cannot determine",
"insufficient evidence",
"need retrieve",
)
_EMPTY_MEMORY_MARKERS ={"","none","No relevant short-term memory","No associated long-term memory","None","none"}


def _normalize_verdict (value :object )->str :
    text =str (value or "").strip ().upper ()
    if text not in {"SUFFICIENT","NEED_RETRIEVE"}:
        return "NEED_RETRIEVE"
    return text 


def _normalize_risk (value :object )->str :
    text =str (value or "").strip ().upper ()
    if text not in {"LOW","MEDIUM","HIGH"}:
        return "MEDIUM"
    return text 


def _clamp_score (value :object ,default :float )->float :
    try :
        score =float (value )
    except (TypeError ,ValueError ):
        return default 
    return max (0.0 ,min (1.0 ,score ))


def _risk_from_score (score :float ,threshold :float )->str :
    if score >=max (0.88 ,threshold +0.12 ):
        return "LOW"
    if score >=threshold :
        return "MEDIUM"
    return "HIGH"


def _is_uncertain_text (value :object )->bool :
    raw =str (value or "").strip ()
    if not raw :
        return True 
    lowered =raw .lower ()
    return any (marker in raw or marker in lowered for marker in _UNCERTAIN_MARKERS )


def _memory_text (value :object )->str :
    return str (value or "").strip ()


def _has_memory_text (value :object )->bool :
    text =_memory_text (value )
    return bool (text and text not in _EMPTY_MEMORY_MARKERS )


def _split_ltm_chunks (long_term_memory :str )->list [str ]:
    if not _has_memory_text (long_term_memory ):
        return []
    parts =[p .strip ()for p in str (long_term_memory ).split ("|")]
    out :list [str ]=[]
    seen :set [str ]=set ()
    for part in parts :
        if not part or part in _EMPTY_MEMORY_MARKERS or part in seen :
            continue 
        seen .add (part )
        out .append (part )
    return out 


def _recent_messages_to_text (recent_messages :list [dict ],keep :int =3 )->str :
    if not recent_messages :
        return ""
    lines :list [str ]=[]
    for msg in recent_messages [-max (1 ,keep ):]:
        role =str (msg .get ("role","")or "").strip ().lower ()or "unknown"
        content =str (msg .get ("content","")or "").strip ()
        if not content :
            continue 
        lines .append (f"{role}: {content}")
    return "\n".join (lines ).strip ()


def _build_memory_verify_state (
state :AgentState ,
effective_query :str ,
draft_answer :str ,
)->AgentState :
    context_pool :list [dict [str ,Any ]]=[]

    ltm_chunks =_split_ltm_chunks (str (state .get ("long_term_memory","")or ""))
    for idx ,chunk in enumerate (ltm_chunks ,start =1 ):
        context_pool .append (
        {
        "id":f"memory_ltm_{idx}",
        "source":"LTM",
        "title":"LTM",
        "module":"LongTermMemory",
        "score":1.0 ,
        "content":chunk ,
        "metadata":{"source":"LTM","h2":"LongTermMemory"},
        }
        )

    short_term_memory =str (state .get ("short_term_memory","")or "")
    if _has_memory_text (short_term_memory ):
        context_pool .append (
        {
        "id":"memory_stm_1",
        "source":"STM",
        "title":"STM",
        "module":"ShortTermMemory",
        "score":1.0 ,
        "content":short_term_memory ,
        "metadata":{"source":"STM","h2":"ShortTermMemory"},
        }
        )

    recent_messages =state .get ("recent_messages",[])or []
    recent_text =_recent_messages_to_text (recent_messages )
    if recent_text :
        context_pool .append (
        {
        "id":"memory_recent_1",
        "source":"RecentMessages",
        "title":"RecentMessages",
        "module":"RecentMessages",
        "score":0.8 ,
        "content":recent_text ,
        "metadata":{"source":"RecentMessages","h2":"RecentMessages"},
        }
        )

    used_chunks =int (state .get ("used_memory_chunks",0 )or 0 )
    cite_top_k =len (context_pool )if used_chunks <=0 else min (used_chunks ,len (context_pool ))
    citations :list [dict [str ,Any ]]=[]
    for doc in context_pool [:cite_top_k ]:
        citations .append (
        {
        "id":str (doc .get ("id","")),
        "title":str (doc .get ("title",doc .get ("source","Memory"))),
        "score":float (doc .get ("score",1.0 )or 1.0 ),
        "quote":str (doc .get ("content","")or "")[:800 ],
        }
        )

    return {
    "query":effective_query ,
    "resolved_query":effective_query ,
    "response":str (draft_answer or ""),
    "citations":citations ,
    "context_pool":context_pool ,
    "evidence_table":[],
    "response_revision":1 ,
    "steps_log":[],
    }


def _run_memory_strict_gate (
state :AgentState ,
effective_query :str ,
draft_answer :str ,
)->dict [str ,Any ]:
    strict_state =_build_memory_verify_state (
    state =state ,
    effective_query =effective_query ,
    draft_answer =draft_answer ,
    )
    out =strict_verify_node (strict_state )
    return {
    "action":str (out .get ("strict_action","REPAIR")).upper (),
    "reason":str (out .get ("strict_reason","")).strip (),
    "score":float (out .get ("strict_score",0.0 )or 0.0 ),
    "total_score":float (out .get ("strict_total_score",0.0 )or 0.0 ),
    "failure_type":str (out .get ("failure_type","")or ""),
    "repair_trigger":str (out .get ("repair_trigger","")or ""),
    "context_count":len (out .get ("context_pool",[])or []),
    "citations_count":len (out .get ("citations",[])or []),
    }


def verify_memory_node (state :AgentState )->AgentState :
    ltm_hits_count =int (state .get ("ltm_hits_count",0 )or 0 )
    used_memory_chunks =int (state .get ("used_memory_chunks",0 )or 0 )
    draft_confidence =float (state .get ("draft_confidence",0.0 )or 0.0 )
    draft_answer =str (state .get ("draft_answer","")or "")
    effective_query =str (state .get ("resolved_query")or state .get ("query","")).strip ()
    prompt_state :AgentState =dict (state )
    prompt_state ["query"]=effective_query 
    threshold =float (AppConfig .MEMORY_SUFFICIENT_SCORE_THRESHOLD )
    strict_gate_enabled =bool (AppConfig .RUNTIME_MEMORY_STRICT_GATE_ENABLED )
    strict_gate_applied =False 
    strict_gate_action =""
    strict_gate_reason =""
    strict_gate_score =0.0 
    strict_gate_total_score =0.0 
    strict_gate_failure_type =""
    strict_gate_trigger =""
    strict_gate_context_count =0 
    strict_gate_citations_count =0 

    shortcut_no_memory =(ltm_hits_count <=0 and used_memory_chunks <=0 )
    shortcut_low_conf =(
    used_memory_chunks <=0 
    and (draft_confidence <0.55 or _is_uncertain_text (draft_answer ))
    )
    model_risk_level ="MEDIUM"

    if shortcut_no_memory :
        memory_score =0.0 
        reason ="no memory evidence: ltm_hits_count=0 and used_memory_chunks=0"
        model_risk_level ="HIGH"
    elif shortcut_low_conf :
        memory_score =min (draft_confidence ,0.45 )
        reason ="memory draft is low confidence or uncertain"
        model_risk_level ="HIGH"
    else :
        out =run_prompt (VerifyMemoryPrompt ,prompt_state )
        llm_verdict =_normalize_verdict (out .get ("verdict"))
        default_score =0.85 if llm_verdict =="SUFFICIENT"else 0.35 
        memory_score =_clamp_score (out .get ("score"),default =default_score )
        reason =str (out .get ("reason","")).strip ()
        model_risk_level =_normalize_risk (out .get ("risk_level"))

    verdict ="SUFFICIENT"if memory_score >=threshold else "NEED_RETRIEVE"
    risk_level =_risk_from_score (memory_score ,threshold )
    risk_rank ={"LOW":0 ,"MEDIUM":1 ,"HIGH":2 }
    if risk_rank [model_risk_level ]>risk_rank [risk_level ]:
        risk_level =model_risk_level 
    if verdict =="NEED_RETRIEVE"and risk_level !="HIGH":
        risk_level ="HIGH"

        # General strict gating (regardless of question type):
        # When the memory is initially judged as SUFFICIENT, a structured verification is performed according to the strict_verify logic;
        # Any failure will fall back to the retrieval path (NEED_RETRIEVE) and no repair will be done in the memory stage.
    if strict_gate_enabled and verdict =="SUFFICIENT":
        strict_gate_applied =True 
        try :
            strict_gate =_run_memory_strict_gate (
            state =state ,
            effective_query =effective_query ,
            draft_answer =draft_answer ,
            )
            strict_gate_action =str (strict_gate .get ("action","REPAIR")).upper ()
            strict_gate_reason =str (strict_gate .get ("reason","")).strip ()
            strict_gate_score =float (strict_gate .get ("score",0.0 )or 0.0 )
            strict_gate_total_score =float (strict_gate .get ("total_score",0.0 )or 0.0 )
            strict_gate_failure_type =str (strict_gate .get ("failure_type","")or "")
            strict_gate_trigger =str (strict_gate .get ("repair_trigger","")or "")
            strict_gate_context_count =int (strict_gate .get ("context_count",0 )or 0 )
            strict_gate_citations_count =int (strict_gate .get ("citations_count",0 )or 0 )
        except Exception as e :# pragma: no cover
            strict_gate_action ="REPAIR"
            strict_gate_reason =f"memory_strict_gate_exception: {type(e).__name__}: {e}"

        if strict_gate_action !="PASS":
            verdict ="NEED_RETRIEVE"
            memory_score =max (0.0 ,min (memory_score ,max (0.0 ,threshold -1e-6 )))
            if strict_gate_reason :
                reason =f"{reason}; memory_strict_gate={strict_gate_reason}"
            risk_level ="HIGH"

    state ["memory_score"]=memory_score 
    state ["memory_verdict"]=verdict 
    state ["memory_reason"]=reason 
    state ["memory_risk_level"]=risk_level 

    state .setdefault ("steps_log",[]).append (
    StepLog (
    node ="verify_memory",
    info ={
    "state":{
    "query_preview":clip_text (state .get ("query",""),180 ),
    "effective_query_preview":clip_text (effective_query ,180 ),
    "draft_confidence":state .get ("draft_confidence",0.0 ),
    "used_memory_chunks":state .get ("used_memory_chunks",0 ),
    "ltm_hits_count":ltm_hits_count ,
    },
    "llm_output":{
    "score":memory_score ,
    "score_threshold":threshold ,
    "verdict":verdict ,
    "risk_level":risk_level ,
    "reason_preview":clip_text (reason ,220 ),
    "shortcut_no_memory":shortcut_no_memory ,
    "shortcut_low_conf":shortcut_low_conf ,
    "strict_gate_enabled":strict_gate_enabled ,
    "strict_gate_applied":strict_gate_applied ,
    "strict_gate_action":strict_gate_action ,
    "strict_gate_score":strict_gate_score ,
    "strict_gate_total_score":strict_gate_total_score ,
    "strict_gate_failure_type":strict_gate_failure_type ,
    "strict_gate_trigger":strict_gate_trigger ,
    "strict_gate_context_count":strict_gate_context_count ,
    "strict_gate_citations_count":strict_gate_citations_count ,
    "strict_gate_reason_preview":clip_text (strict_gate_reason ,220 ),
    },
    },
    timestamp =time .time (),
    )
    )
    return state 
