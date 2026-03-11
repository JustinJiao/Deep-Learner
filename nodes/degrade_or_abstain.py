import time

from core.state import AgentState, StepLog
from nodes.log_utils import clip_text


def _collect_missing_evidence_hints(state: AgentState) -> list[str]:
    hints: list[str] = []
    route_facts = state.get("route_facts", []) or []
    for fact in route_facts:
        if not bool(fact.get("missing", False)):
            continue
        entity = str(fact.get("entity", "")).strip()
        route_query = str(fact.get("route_query", "")).strip()
        reason = str(fact.get("missing_reason", "")).strip() or "insufficient_route_evidence"
        label = entity or route_query or f"route_{fact.get('route_index', '?')}"
        hints.append(f"{label}: {reason}")

    coverage = state.get("route_fact_coverage", {}) or {}
    missing_entities = list(coverage.get("missing_entities", []) or [])
    for entity in missing_entities:
        text = str(entity or "").strip()
        if text:
            hints.append(f"{text}: missing_entity_coverage")

    repair_trigger = str(state.get("repair_trigger", "")).strip()
    if repair_trigger and not hints:
        hints.append(f"trigger={repair_trigger}")

    deduped: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        key = hint.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hint)
    return deduped


def degrade_or_abstain_node(state: AgentState) -> AgentState:
    reason = state.get("strict_reason", "") or state.get("memory_reason", "")
    failure_type = state.get("failure_type", "")
    repair_trigger = state.get("repair_trigger", "")
    last_compose_response = str(state.get("response", "") or state.get("previous_response", "")).strip()
    if not last_compose_response:
        last_compose_response = "Uncertain."

    hints = _collect_missing_evidence_hints(state)
    missing_line = "；".join(hints[:6]) if hints else "未识别到可定位缺口"

    answer = (
        "当前证据存在缺口或冲突，以下为可定位的缺失证据提示：\n"
        f"{missing_line}\n\n"
        "以下是最后一次 compose 的候选回答（仅供参考，未通过严格校验）：\n"
        f"{last_compose_response}"
    )
    if reason:
        answer += f"\n\n严格校验信息：{reason}"
    if failure_type:
        answer += f"\n失败类型：{failure_type}"
    if repair_trigger:
        answer += f"\n触发规则：{repair_trigger}"

    state["response"] = answer
    state["citations"] = state.get("citations", []) or []
    state["run_status"] = "degraded"
    state["strict_status"] = "FAILED"

    state.setdefault("steps_log", []).append(
        StepLog(
            node="degrade_or_abstain",
            info={
                "state": {
                    "strict_reason_preview": clip_text(reason, 220),
                    "failure_type": failure_type,
                    "repair_trigger": repair_trigger,
                    "response_preview": clip_text(answer, 220),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
