# core/debug_executor.py

from __future__ import annotations 
import copy 
import json 
from typing import Any 

from core .registry import NODE_REGISTRY 
from core .errors import UnknownNodeError 
from core .state import AgentState ,StepLog 
from session .store import get_session 


def pretty (obj ):
    print (json .dumps (obj ,indent =2 ,ensure_ascii =False ,default =str ))


def diff_state (prev ,curr ):
    changed ={}
    for k in curr :
        if k not in prev or prev [k ]!=curr [k ]:
            changed [k ]=curr [k ]
    return changed 


class DebugAgentExecutor :
    """Executor dedicated to observable debugging
    - Do not swallow exceptions
    - Print state diff before and after each node
    - Print session STM changes"""

    def run (self ,session_id :str ,query :str )->AgentState :
        state :AgentState ={
        "session_id":session_id ,
        "query":query ,
        "steps_log":[],
        "loop_count":0 ,
        "run_status":"running",
        }

        print ("\n"+"="*80 )
        print ("DEBUG EXECUTION START")
        print ("="*80 )

        ctx_before =copy .deepcopy (get_session (session_id ).stm )
        print ("\nSESSION BEFORE:")
        pretty (ctx_before )

        prev_state ={}

        # ---------------------
        # fixed preceding node
        # ---------------------
        for step in ["stm_read","intent","planner"]:
            print ("\n"+"-"*80 )
            print (f"[STEP] {step}")
            print ("-"*80 )

            state =NODE_REGISTRY [step ](state )

            delta =diff_state (prev_state ,state )
            print ("STATE DELTA:")
            pretty (delta )

            prev_state =copy .deepcopy (state )

        plan =state ["plan"]

        # ---------------------
        # main loop
        # ---------------------
        while not plan .is_finished ():
            step =plan .current_step ()

            print ("\n"+"-"*80 )
            print (f"[STEP] {step}")
            print ("-"*80 )

            node_fn =NODE_REGISTRY .get (step )
            if node_fn is None :
                raise UnknownNodeError (step )

            state =node_fn (state )

            delta =diff_state (prev_state ,state )
            print ("STATE DELTA:")
            pretty (delta )

            prev_state =copy .deepcopy (state )

            if step =="verify"and state .get ("is_hallucination",False ):
                state ["loop_count"]+=1 
                print (f"VERIFY FAIL → loop_count={state['loop_count']}")
                state =NODE_REGISTRY ["repair"](state )
                continue 

            plan .advance ()

        state ["run_status"]="ok"

        # ---------------------
        # Finalize
        # ---------------------
        for tail in ["finalize","stm_write","stm_summary","persist_ltm"]:
            print ("\n"+"-"*80 )
            print (f"[TAIL] {tail}")
            print ("-"*80 )

            state =NODE_REGISTRY [tail ](state )

            delta =diff_state (prev_state ,state )
            print ("STATE DELTA:")
            pretty (delta )

            prev_state =copy .deepcopy (state )

        print ("\n"+"="*80 )
        print ("FINAL STATE")
        print ("="*80 )
        pretty (state )

        ctx_after =get_session (session_id ).stm 
        print ("\nSESSION AFTER:")
        pretty (ctx_after )

        print ("\n"+"="*80 )
        print ("DEBUG EXECUTION END")
        print ("="*80 )

        return state 
