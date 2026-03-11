from __future__ import annotations

import copy
from typing import Any

from config.settings import AppConfig
from core.errors import (
    InvalidStateTransitionError,
    NodeContractViolationError,
    RuntimeMaxTransitionExceededError,
    UnknownNodeError,
)
from core.registry import NODE_CONTRACTS, NODE_REGISTRY, validate_node_contract
from core.state import AgentState, RuntimeStage, StepLog, build_initial_state


STAGE_ORDER: dict[RuntimeStage, int] = {
    "MEMORY": 0,
    "PHASE1": 1,
    "PHASE2": 2,
    "REPAIR": 3,
    "DEGRADE": 4,
    "FINALIZE": 5,
}

ALLOWED_TRANSITIONS: dict[RuntimeStage, set[RuntimeStage]] = {
    "MEMORY": {"PHASE1", "FINALIZE"},
    "PHASE1": {"PHASE2", "REPAIR", "DEGRADE", "FINALIZE"},
    "PHASE2": {"REPAIR", "DEGRADE", "FINALIZE"},
    "REPAIR": {"DEGRADE", "FINALIZE"},
    "DEGRADE": {"FINALIZE"},
    "FINALIZE": set(),
}


def _append_log(state: AgentState, log: StepLog) -> None:
    logs = state.setdefault("steps_log", [])
    logs.append(log)
    cap = int(AppConfig.MAX_STEPS_LOG)
    if cap > 0 and len(logs) > cap:
        state["steps_log"] = logs[-cap:]


def _assert_forward_only(from_stage: RuntimeStage, to_stage: RuntimeStage) -> None:
    if STAGE_ORDER[to_stage] < STAGE_ORDER[from_stage]:
        raise InvalidStateTransitionError(
            f"backward transition is forbidden: {from_stage} -> {to_stage}"
        )

    allowed = ALLOWED_TRANSITIONS.get(from_stage, set())
    if to_stage not in allowed:
        raise InvalidStateTransitionError(
            f"transition not allowed: {from_stage} -> {to_stage}"
        )


def _assert_not_exceed_max_transitions(state: AgentState) -> None:
    max_transitions = int(AppConfig.RUNTIME_MAX_TRANSITIONS)
    if state.get("transition_count", 0) > max_transitions:
        raise RuntimeMaxTransitionExceededError(
            f"runtime transition count exceeded: {state.get('transition_count', 0)} > {max_transitions}"
        )


def _guard_phase2_once(state: AgentState) -> None:
    if state.get("phase2_used", False):
        raise InvalidStateTransitionError("retrieve_phase2 can only run once")


def _guard_repair_once(state: AgentState) -> None:
    if state.get("repair_used", False):
        raise InvalidStateTransitionError("repair can only run once")


def _transition(state: AgentState, to_stage: RuntimeStage, reason: str) -> None:
    from_stage = state.get("runtime_stage", "MEMORY")
    _assert_forward_only(from_stage, to_stage)

    state["runtime_stage"] = to_stage
    state["transition_count"] = state.get("transition_count", 0) + 1
    _assert_not_exceed_max_transitions(state)

    _append_log(
        state,
        StepLog(
            node="executor",
            info={
                "runtime_stage_before": from_stage,
                "runtime_stage_after": to_stage,
                "transition_reason": reason,
                "transition_count": state.get("transition_count", 0),
                "phase2_used": state.get("phase2_used", False),
                "repair_used": state.get("repair_used", False),
                "repair_mode": state.get("repair_mode", False),
            },
        ),
    )


class AgentExecutor:
    """生产级单轮执行入口（具备降级与防污染写入）。"""

    def run(self, session_id: str, query: str) -> AgentState:
        state = build_initial_state(session_id=session_id, query=query)
        if AppConfig.RUNTIME_V2_ENABLED:
            return self._run_v2_state_machine(state)
        return self._run_legacy_flow(state)

    def _execute_node(self, state: AgentState, node_name: str) -> AgentState:
        node_fn = NODE_REGISTRY.get(node_name)
        if node_fn is None:
            raise UnknownNodeError(node_name)

        before_state = copy.deepcopy(state)
        out = node_fn(state)

        if AppConfig.RUNTIME_ENFORCE_CONTRACT:
            report = validate_node_contract(
                node_name=node_name,
                before_state=before_state,
                after_state=out,
                contracts=NODE_CONTRACTS,
            )
            if report["enforced"] and not report["valid"]:
                raise NodeContractViolationError(
                    f"{node_name} contract violated: "
                    f"missing_reads={report['missing_reads']} "
                    f"unexpected_writes={report['unexpected_writes']}"
                )

        return out

    def _run_legacy_flow(self, state: AgentState) -> AgentState:
        try:
            state = self._execute_node(state, "stm_read")
            state = self._execute_node(state, "intent")
            state = self._execute_node(state, "planner")

            plan = state["plan"]

            while not plan.is_finished():
                step = plan.current_step()
                state = self._execute_node(state, step)

                if step == "verify" and state.get("is_hallucination", False):
                    state["loop_count"] = state.get("loop_count", 0) + 1
                    _append_log(
                        state,
                        StepLog(node="executor", info=f"verify_fail loop={state['loop_count']}"),
                    )

                    if state["loop_count"] > plan.max_loops:
                        state["run_status"] = "degraded"
                        state["response"] = (
                            "我刚才的回答缺少可靠依据，无法在当前资料范围内自洽验证。\n"
                            "为了避免误导，请你补充：你希望我基于哪些具体文档/数据，或提供更多上下文。\n"
                            "（我也可以先给一个不确定但可能的方向，并明确标注假设。）"
                        )
                        state["is_hallucination"] = False
                        _append_log(
                            state,
                            StepLog(node="executor", info="max_loops reached -> degraded response"),
                        )
                        plan.finish()
                        break

                    state = self._execute_node(state, "repair")
                    continue

                plan.advance()

            if state.get("run_status") == "running":
                state["run_status"] = "ok"

        except Exception as e:
            state["run_status"] = "error"
            state["error"] = {"type": type(e).__name__, "message": str(e)}
            _append_log(state, StepLog(node="executor", info=f"exception: {type(e).__name__}: {e}"))

            if not state.get("response"):
                state["response"] = "系统在处理时发生错误，我无法完成本次回答。请重试或提供更多信息。"

        finally:
            state = self._run_tail_nodes(state)

        return state

    def _run_v2_state_machine(self, state: AgentState) -> AgentState:
        try:
            state = self._execute_node(state, "stm_read")
            # 先做一次 LTM 召回（基于原 query），给指代消解提供背景锚点。
            state = self._execute_node(state, "ltm_recall")
            state = self._execute_node(state, "resolve_query_reference")
            # 指代消解后再次召回，刷新下游 memory/retrieval 使用的长期记忆上下文。
            state = self._execute_node(state, "ltm_recall")
            state = self._execute_node(state, "compose_memory_draft")
            state = self._execute_node(state, "verify_memory")

            memory_sufficient = state.get("memory_verdict") == "SUFFICIENT"
            force_retrieve = bool(AppConfig.RUNTIME_FORCE_RETRIEVE_WHEN_MEMORY_SUFFICIENT)
            if memory_sufficient and not force_retrieve:
                state["response"] = state.get("draft_answer", "")
                state.setdefault("citations", [])
                state["run_status"] = "ok"
                _transition(state, "FINALIZE", reason="memory_verdict_sufficient")
            else:
                phase1_reason = (
                    "memory_verdict_sufficient_but_force_retrieve"
                    if memory_sufficient and force_retrieve
                    else "memory_verdict_need_retrieve"
                )
                _transition(state, "PHASE1", reason=phase1_reason)
                state = self._execute_node(state, "rewrite_query_for_retrieval")
                state = self._execute_node(state, "retrieve_phase1")
                state = self._execute_node(state, "rerank_phase1")
                state = self._execute_node(state, "extract_route_facts")
                state.setdefault("repair_mode", False)
                state.setdefault("strict_reason", "")
                state.setdefault("previous_response", state.get("response", ""))
                state = self._execute_node(state, "compose_with_context")
                state = self._execute_node(state, "strict_verify")

                strict_action = str(state.get("strict_action", "PASS")).upper()
                if strict_action == "PASS":
                    state["run_status"] = "ok"
                    state["strict_status"] = "PASS"
                    _transition(state, "FINALIZE", reason="strict_verify_pass")
                else:
                    failure_type = state.get("failure_type", "")
                    if (
                        str(failure_type).upper() == "INSUFFICIENT_EVIDENCE"
                        and not bool(state.get("phase2_used", False))
                    ):
                        _guard_phase2_once(state)
                        state["phase2_used"] = True
                        _transition(
                            state,
                            "PHASE2",
                            reason=f"phase1_fail_expand_retrieval_{state.get('repair_trigger', '') or 'insufficient_evidence'}",
                        )
                        state = self._execute_node(state, "retrieve_phase2")
                        state = self._execute_node(state, "rerank_phase2")
                        state = self._execute_node(state, "extract_route_facts")
                        state.setdefault("repair_mode", False)
                        state.setdefault("strict_reason", "")
                        state.setdefault("previous_response", state.get("response", ""))
                        state = self._execute_node(state, "compose_with_context")
                        state = self._execute_node(state, "strict_verify")
                        strict_action = str(state.get("strict_action", "PASS")).upper()
                        failure_type = state.get("failure_type", "")

                        if strict_action == "PASS":
                            state["run_status"] = "ok"
                            state["strict_status"] = "PASS"
                            _transition(state, "FINALIZE", reason="strict_verify_pass_after_phase2")

                    if (
                        strict_action != "PASS"
                        and str(failure_type).upper() == "INSUFFICIENT_EVIDENCE"
                        and bool(state.get("phase2_used", False))
                    ):
                        # phase2 expanded retrieval already attempted; return partial answer
                        # with explicit missing-evidence notice instead of degrading.
                        state["run_status"] = "ok"
                        state["strict_status"] = "FAILED"
                        _transition(
                            state,
                            "FINALIZE",
                            reason="insufficient_evidence_after_phase2_return_partial",
                        )
                    elif not state.get("repair_used", False):
                        if strict_action != "PASS":
                            _guard_repair_once(state)
                            _transition(
                                state,
                                "REPAIR",
                                reason=f"phase1_fail_{state.get('repair_trigger', '') or str(failure_type).lower()}",
                            )
                            state = self._execute_node(state, "set_repair_mode")
                            state = self._execute_node(state, "extract_route_facts")
                            state.setdefault("previous_response", state.get("response", ""))
                            state = self._execute_node(state, "compose_with_context")
                            state = self._execute_node(state, "strict_verify")
                            strict_action = str(state.get("strict_action", "PASS")).upper()

                            if strict_action == "PASS":
                                state["run_status"] = "ok"
                                state["strict_status"] = "REPAIRED"
                                _transition(state, "FINALIZE", reason="strict_verify_pass_after_repair")
                            else:
                                state["strict_status"] = "FAILED"
                                _transition(
                                    state,
                                    "DEGRADE",
                                    reason=f"strict_verify_fail_after_repair:{state.get('failure_type', '')}",
                                )
                                state = self._execute_node(state, "degrade_or_abstain")
                                _transition(state, "FINALIZE", reason="degrade_after_repair_fail")
                    else:
                        if strict_action != "PASS":
                            state["strict_status"] = "FAILED"
                            _transition(
                                state,
                                "DEGRADE",
                                reason=f"strict_verify_fail_unrecoverable:{failure_type}",
                            )
                            state = self._execute_node(state, "degrade_or_abstain")
                            _transition(state, "FINALIZE", reason="degrade_after_unrecoverable_fail")
        except Exception as e:
            state["run_status"] = "error"
            state["error"] = {"type": type(e).__name__, "message": str(e)}
            _append_log(state, StepLog(node="executor", info=f"v2_exception: {type(e).__name__}: {e}"))
            if not state.get("response"):
                state["response"] = "系统在处理时发生错误，我无法完成本次回答。请重试或提供更多信息。"
        finally:
            state = self._run_tail_nodes(state)
        return state

    def _run_tail_nodes(self, state: AgentState) -> AgentState:
        should_persist = bool(state.get("response")) and state.get("run_status") in ("ok", "degraded")
        ltm_write_enabled = bool(AppConfig.LTM_WRITE_ENABLED)
        for tail in ("finalize", "stm_write", "stm_summary", "persist_ltm"):
            if tail in ("stm_write", "stm_summary", "persist_ltm") and not should_persist:
                continue
            if tail == "persist_ltm" and not ltm_write_enabled:
                _append_log(state, StepLog(node=tail, info="skipped: LTM_WRITE_ENABLED=false"))
                continue
            try:
                state = self._execute_node(state, tail)
            except Exception as e:
                _append_log(state, StepLog(node=tail, info=f"error: {type(e).__name__}: {e}"))
        return state
