from __future__ import annotations 

import copy 

from config .settings import AppConfig 
from core .errors import (
InvalidStateTransitionError ,
NodeContractViolationError ,
RuntimeMaxTransitionExceededError ,
UnknownNodeError ,
)
from core .registry import NODE_CONTRACTS ,NODE_REGISTRY ,validate_node_contract 
from core .state import AgentState ,RuntimeStage ,StepLog ,build_initial_state 


STAGE_ORDER :dict [RuntimeStage ,int ]={
"MEMORY":0 ,
"PHASE1":1 ,
"PHASE2":2 ,
"REPAIR":3 ,
"DEGRADE":4 ,
"FINALIZE":5 ,
}

ALLOWED_TRANSITIONS :dict [RuntimeStage ,set [RuntimeStage ]]={
"MEMORY":{"PHASE1","FINALIZE"},
"PHASE1":{"PHASE2","REPAIR","DEGRADE","FINALIZE"},
"PHASE2":{"REPAIR","DEGRADE","FINALIZE"},
"REPAIR":{"DEGRADE","FINALIZE"},
"DEGRADE":{"FINALIZE"},
"FINALIZE":set (),
}


def _append_log (state :AgentState ,log :StepLog )->None :
    logs =state .setdefault ("steps_log",[])
    logs .append (log )
    cap =int (AppConfig .MAX_STEPS_LOG )
    if cap >0 and len (logs )>cap :
        state ["steps_log"]=logs [-cap :]


def _assert_forward_only (from_stage :RuntimeStage ,to_stage :RuntimeStage )->None :
    if STAGE_ORDER [to_stage ]<STAGE_ORDER [from_stage ]:
        raise InvalidStateTransitionError (
        f"backward transition is forbidden: {from_stage} -> {to_stage}"
        )

    allowed =ALLOWED_TRANSITIONS .get (from_stage ,set ())
    if to_stage not in allowed :
        raise InvalidStateTransitionError (
        f"transition not allowed: {from_stage} -> {to_stage}"
        )


def _assert_not_exceed_max_transitions (state :AgentState )->None :
    max_transitions =int (AppConfig .RUNTIME_MAX_TRANSITIONS )
    if state .get ("transition_count",0 )>max_transitions :
        raise RuntimeMaxTransitionExceededError (
        f"runtime transition count exceeded: {state.get('transition_count', 0)} > {max_transitions}"
        )


def _guard_phase2_once (state :AgentState )->None :
    if state .get ("phase2_used",False ):
        raise InvalidStateTransitionError ("retrieve_phase2 can only run once")


def _guard_repair_once (state :AgentState )->None :
    if state .get ("repair_used",False ):
        raise InvalidStateTransitionError ("repair can only run once")


def _transition (state :AgentState ,to_stage :RuntimeStage ,reason :str )->None :
    from_stage =state .get ("runtime_stage","MEMORY")
    _assert_forward_only (from_stage ,to_stage )

    state ["runtime_stage"]=to_stage 
    state ["transition_count"]=state .get ("transition_count",0 )+1 
    _assert_not_exceed_max_transitions (state )

    _append_log (
    state ,
    StepLog (
    node ="executor",
    info ={
    "runtime_stage_before":from_stage ,
    "runtime_stage_after":to_stage ,
    "transition_reason":reason ,
    "transition_count":state .get ("transition_count",0 ),
    "phase2_used":state .get ("phase2_used",False ),
    "repair_used":state .get ("repair_used",False ),
    "repair_mode":state .get ("repair_mode",False ),
    },
    ),
    )


class AgentExecutor :
    """Production-grade single-round execution entry (with downgrade and anti-pollution writing)."""

    def run (self ,session_id :str ,query :str )->AgentState :
        state =build_initial_state (session_id =session_id ,query =query )
        if not AppConfig .RUNTIME_V2_ENABLED :
            raise RuntimeError ("Legacy runtime flow has been removed; set RUNTIME_V2_ENABLED=true")
        return self ._run_v2_state_machine (state )

    def _execute_node (self ,state :AgentState ,node_name :str )->AgentState :
        node_fn =NODE_REGISTRY .get (node_name )
        if node_fn is None :
            raise UnknownNodeError (node_name )

        before_state =copy .deepcopy (state )
        out =node_fn (state )

        if AppConfig .RUNTIME_ENFORCE_CONTRACT :
            report =validate_node_contract (
            node_name =node_name ,
            before_state =before_state ,
            after_state =out ,
            contracts =NODE_CONTRACTS ,
            )
            if report ["enforced"]and not report ["valid"]:
                raise NodeContractViolationError (
                f"{node_name} contract violated: "
                f"missing_reads={report['missing_reads']} "
                f"unexpected_writes={report['unexpected_writes']}"
                )

        return out 

    def _compose_and_strict_verify (
        self ,
        state :AgentState ,
        *,
        reset_repair_defaults :bool =False ,
    )->tuple [AgentState ,str ]:
        if reset_repair_defaults :
            state .setdefault ("repair_mode",False )
            state .setdefault ("strict_reason","")
        state .setdefault ("previous_response",state .get ("response",""))
        state =self ._execute_node (state ,"compose_with_context")
        state =self ._execute_node (state ,"strict_verify")
        strict_action =str (state .get ("strict_action","PASS")).upper ()
        return state ,strict_action 

    def _run_v2_state_machine (self ,state :AgentState )->AgentState :
        try :
            state =self ._execute_node (state ,"stm_read")
            # First do an LTM recall (based on the original query) to provide background anchor points for reference resolution.
            state =self ._execute_node (state ,"ltm_recall")
            state =self ._execute_node (state ,"resolve_query_reference")
            # Recalling again after reference resolution refreshes the long-term memory context used by downstream memory/retrieval.
            state =self ._execute_node (state ,"ltm_recall")
            state =self ._execute_node (state ,"compose_memory_draft")
            state =self ._execute_node (state ,"verify_memory")

            memory_sufficient =state .get ("memory_verdict")=="SUFFICIENT"
            force_retrieve =bool (AppConfig .RUNTIME_FORCE_RETRIEVE_WHEN_MEMORY_SUFFICIENT )
            if memory_sufficient and not force_retrieve :
                state ["response"]=state .get ("draft_answer","")
                state .setdefault ("citations",[])
                state ["run_status"]="ok"
                _transition (state ,"FINALIZE",reason ="memory_verdict_sufficient")
            else :
                phase1_reason =(
                "memory_verdict_sufficient_but_force_retrieve"
                if memory_sufficient and force_retrieve 
                else "memory_verdict_need_retrieve"
                )
                _transition (state ,"PHASE1",reason =phase1_reason )
                state =self ._execute_node (state ,"rewrite_query_for_retrieval")
                state =self ._execute_node (state ,"retrieve_phase1")
                state =self ._execute_node (state ,"rerank_phase1")
                state =self ._execute_node (state ,"extract_route_facts")
                state ,strict_action =self ._compose_and_strict_verify (
                state ,
                reset_repair_defaults =True ,
                )
                if strict_action =="PASS":
                    state ["run_status"]="ok"
                    state ["strict_status"]="PASS"
                    _transition (state ,"FINALIZE",reason ="strict_verify_pass")
                else :
                    failure_type =state .get ("failure_type","")
                    if (
                    str (failure_type ).upper ()=="INSUFFICIENT_EVIDENCE"
                    and not bool (state .get ("phase2_used",False ))
                    ):
                        _guard_phase2_once (state )
                        state ["phase2_used"]=True 
                        _transition (
                        state ,
                        "PHASE2",
                        reason =f"phase1_fail_expand_retrieval_{state.get('repair_trigger', '') or 'insufficient_evidence'}",
                        )
                        state =self ._execute_node (state ,"retrieve_phase2")
                        state =self ._execute_node (state ,"rerank_phase2")
                        state =self ._execute_node (state ,"extract_route_facts")
                        state ,strict_action =self ._compose_and_strict_verify (
                        state ,
                        reset_repair_defaults =True ,
                        )
                        failure_type =state .get ("failure_type","")

                        if strict_action =="PASS":
                            state ["run_status"]="ok"
                            state ["strict_status"]="PASS"
                            _transition (state ,"FINALIZE",reason ="strict_verify_pass_after_phase2")

                    if (
                    strict_action !="PASS"
                    and str (failure_type ).upper ()=="INSUFFICIENT_EVIDENCE"
                    and bool (state .get ("phase2_used",False ))
                    ):
                    # phase2 expanded retrieval already attempted; return partial answer
                    # with explicit missing-evidence notice instead of degrading.
                        state ["run_status"]="ok"
                        state ["strict_status"]="FAILED"
                        _transition (
                        state ,
                        "FINALIZE",
                        reason ="insufficient_evidence_after_phase2_return_partial",
                        )
                    elif not state .get ("repair_used",False ):
                        if strict_action !="PASS":
                            _guard_repair_once (state )
                            _transition (
                            state ,
                            "REPAIR",
                            reason =f"phase1_fail_{state.get('repair_trigger', '') or str(failure_type).lower()}",
                            )
                            state =self ._execute_node (state ,"set_repair_mode")
                            state =self ._execute_node (state ,"extract_route_facts")
                            state ,strict_action =self ._compose_and_strict_verify (state )

                            if strict_action =="PASS":
                                state ["run_status"]="ok"
                                state ["strict_status"]="REPAIRED"
                                _transition (state ,"FINALIZE",reason ="strict_verify_pass_after_repair")
                            else :
                                state ["strict_status"]="FAILED"
                                _transition (
                                state ,
                                "DEGRADE",
                                reason =f"strict_verify_fail_after_repair:{state.get('failure_type', '')}",
                                )
                                state =self ._execute_node (state ,"degrade_or_abstain")
                                _transition (state ,"FINALIZE",reason ="degrade_after_repair_fail")
                    else :
                        if strict_action !="PASS":
                            state ["strict_status"]="FAILED"
                            _transition (
                            state ,
                            "DEGRADE",
                            reason =f"strict_verify_fail_unrecoverable:{failure_type}",
                            )
                            state =self ._execute_node (state ,"degrade_or_abstain")
                            _transition (state ,"FINALIZE",reason ="degrade_after_unrecoverable_fail")
        except Exception as e :
            state ["run_status"]="error"
            state ["error"]={"type":type (e ).__name__ ,"message":str (e )}
            _append_log (state ,StepLog (node ="executor",info =f"v2_exception: {type(e).__name__}: {e}"))
            if not state .get ("response"):
                state ["response"]="The system encountered an error during processing and I cannot complete this answer. Please try again or provide more information."
        finally :
            state =self ._run_tail_nodes (state )
        return state 

    def _run_tail_nodes (self ,state :AgentState )->AgentState :
        should_persist =bool (state .get ("response"))and state .get ("run_status")in ("ok","degraded")
        ltm_write_enabled =bool (AppConfig .LTM_WRITE_ENABLED )
        for tail in ("finalize","stm_write","stm_summary","persist_ltm"):
            if tail in ("stm_write","stm_summary","persist_ltm")and not should_persist :
                continue 
            if tail =="persist_ltm"and not ltm_write_enabled :
                _append_log (state ,StepLog (node =tail ,info ="skipped: LTM_WRITE_ENABLED=false"))
                continue 
            try :
                state =self ._execute_node (state ,tail )
            except Exception as e :
                _append_log (state ,StepLog (node =tail ,info =f"error: {type(e).__name__}: {e}"))
        return state 
