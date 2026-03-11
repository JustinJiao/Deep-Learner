# nodes/stm_read.py

import time 

from session .store import get_session 
from memory .stm import STM 
from core .state import AgentState ,StepLog 
from nodes .log_utils import clip_text ,preview_messages ,preview_turns 


def stm_read_node (state :AgentState )->AgentState :
    ctx =get_session (state ["session_id"])

    if not isinstance (ctx .stm ,dict ):
        raise TypeError ("ctx.stm must be dict.")

    stm =STM (ctx .stm )

    # --------------------------
    # summary to LLM
    # --------------------------
    state ["short_term_memory"]=stm .get_summary_text ()

    # --------------------------
    # turn -> message conversion
    # --------------------------
    messages_for_llm =[]

    for turn in stm .recent_messages :
        if turn .get ("query"):
            messages_for_llm .append ({
            "role":"user",
            "content":turn ["query"]
            })
        if turn .get ("response"):
            messages_for_llm .append ({
            "role":"assistant",
            "content":turn ["response"]
            })

    state ["recent_messages"]=messages_for_llm 

    state .setdefault ("steps_log",[]).append (
    StepLog (
    node ="stm_read",
    info ={
    "memory":{
    "summary_preview":clip_text (state .get ("short_term_memory",""),180 ),
    "recent_turns_preview":preview_turns (stm .recent_messages ),
    "messages_count":len (stm .messages ),
    "summary_blocks":len (stm .summary ),
    },
    "state":{
    "recent_messages_for_llm_preview":preview_messages (messages_for_llm ),
    },
    },
    timestamp =time .time (),
    )
    )

    return state 
