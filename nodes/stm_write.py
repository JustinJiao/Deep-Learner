# nodes/stm_write.py

import time 

from session .store import get_session ,save_session 
from memory .stm import STM 
from core .state import AgentState ,StepLog 
from nodes .log_utils import clip_text ,preview_turns 


def stm_write_node (state :AgentState )->AgentState :
    ctx =get_session (state ["session_id"])

    if not isinstance (ctx .stm ,dict ):
        raise TypeError ("ctx.stm must be dict.")

    stm =STM (ctx .stm )
    messages_count_before =len (stm .messages )
    recent_count_before =len (stm .recent_messages )

    query =state .get ("query")
    response =state .get ("response")
    wrote_turn =bool (query and response )

    if wrote_turn :
        stm .append_turn (query ,response )

        # update recent window
    stm .update_recent_messages (window_size =3 )
    messages_count_after =len (stm .messages )
    recent_count_after =len (stm .recent_messages )

    save_session (state ["session_id"],ctx )

    state .setdefault ("steps_log",[]).append (
    StepLog (
    node ="stm_write",
    info ={
    "state":{
    "wrote_turn":wrote_turn ,
    "query_preview":clip_text (query ,160 ),
    "response_preview":clip_text (response ,180 ),
    },
    "memory":{
    "messages_count_before":messages_count_before ,
    "messages_count_after":messages_count_after ,
    "recent_messages_count_before":recent_count_before ,
    "recent_messages_count_after":recent_count_after ,
    "recent_turns_preview":preview_turns (stm .recent_messages ),
    },
    },
    timestamp =time .time (),
    )
    )

    return state 
