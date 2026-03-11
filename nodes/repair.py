# nodes/repair.py
import time 

from core .state import AgentState ,StepLog 
from nodes .log_utils import clip_text 


def repair_node (state :AgentState )->AgentState :
    """Perform "route repair" based on the error type and next_step given by verify.

    - There is no problem with retrieval but there is a problem with generation -> Return to compose (reuse context_pool)
    - Insufficient documentation -> back to retrieve (empty old context_pool)
    - Wrong search direction/ambiguity -> return to query_rewrite (clear old context_pool)"""
    critique =state .get ("critique")or {}
    next_step =critique .get ("next_step","compose")
    error_type =critique .get ("error_type","UNKNOWN")
    hint =critique .get ("critique","")

    # Repair tips for compose
    state ["repair_hint"]=f"[{error_type}] {hint}".strip ()

    plan =state ["plan"]
    before_step_idx =plan .step_idx 
    before_step_name =None if plan .is_finished ()else plan .current_step ()
    cleared_context_pool =False 

    # Routing + Defense
    if next_step =="compose":
        plan .jump_to ("compose")
    elif next_step =="retrieve":
    # Empty the old document pool before re-retrieving to avoid reusing expired results.
        state .pop ("context_pool",None )
        cleared_context_pool =True 
        plan .jump_to ("retrieve")
    elif next_step =="query_rewrite":
        state .pop ("context_pool",None )
        cleared_context_pool =True 
        plan .jump_to ("query_rewrite")
    else :
        plan .jump_to ("compose")
        next_step ="compose"

    state .setdefault ("steps_log",[]).append (
    StepLog (
    node ="repair",
    info ={
    "state":{
    "loop_count":state .get ("loop_count",0 ),
    "plan_before":{
    "step_idx":before_step_idx ,
    "current_step":before_step_name ,
    },
    "plan_after":{
    "step_idx":plan .step_idx ,
    "current_step":None if plan .is_finished ()else plan .current_step (),
    },
    },
    "decision":{
    "error_type":error_type ,
    "next_step_requested":critique .get ("next_step"),
    "next_step_applied":next_step ,
    "context_pool_cleared":cleared_context_pool ,
    "repair_hint_preview":clip_text (state .get ("repair_hint",""),180 ),
    },
    },
    timestamp =time .time (),
    )
    )
    return state 
