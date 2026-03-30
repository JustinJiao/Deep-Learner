# core/debug_executor.py

from __future__ import annotations 
import copy 
import json 

from core .executor import AgentExecutor 
from core .state import AgentState 
from session .store import get_session 


def pretty (obj ):
    print (json .dumps (obj ,indent =2 ,ensure_ascii =False ,default =str ))


class DebugAgentExecutor :
    """V2 debugging wrapper for the production executor."""

    def run (self ,session_id :str ,query :str )->AgentState :
        print ("\n"+"="*80 )
        print ("DEBUG EXECUTION START")
        print ("="*80 )

        ctx_before =copy .deepcopy (get_session (session_id ).stm )
        print ("\nSESSION BEFORE:")
        pretty (ctx_before )

        state =AgentExecutor ().run (session_id =session_id ,query =query )

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
