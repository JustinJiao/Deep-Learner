from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from nodes.intent import intent_node
from nodes.planner import planner_node
from nodes.recall_ltm import recall_ltm_node
from nodes.query_rewrite import query_rewrite_node
from nodes.retrieve import retrieve_node
from nodes.compose import compose_node
from nodes.verify import verify_node
from nodes.repair import repair_node
from nodes.persist_ltm import persist_ltm_node
from nodes.finalize import finalize_node
from nodes.stm_read import stm_read_node
from nodes.stm_write import stm_write_node
from nodes.stm_summary import stm_summary_node
from nodes.ltm_recall import ltm_recall_node
from nodes.resolve_query_reference import resolve_query_reference_node
from nodes.rewrite_query_for_retrieval import rewrite_query_for_retrieval_node
from nodes.compose_memory_draft import compose_memory_draft_node
from nodes.verify_memory import verify_memory_node
from nodes.retrieve_phase1 import retrieve_phase1_node
from nodes.rerank_phase1 import rerank_phase1_node
from nodes.compose_with_context import compose_with_context_node
from nodes.strict_verify import strict_verify_node
from nodes.degrade_or_abstain import degrade_or_abstain_node
from nodes.retrieve_phase2 import retrieve_phase2_node
from nodes.rerank_phase2 import rerank_phase2_node
from nodes.set_repair_mode import set_repair_mode_node


@dataclass(frozen=True)
class NodeContract:
    name: str
    reads: set[str] = field(default_factory=set)
    writes: set[str] = field(default_factory=set)
    llm_node: bool = False
    # PR1 默认以非严格模式接入，避免影响既有链路；后续 PR 可逐步开启 strict=True
    strict: bool = False


NODE_REGISTRY: dict[str, Callable[..., Any]] = {
    "stm_read": stm_read_node,
    "intent": intent_node,
    "planner": planner_node,
    "recall_ltm": recall_ltm_node,
    "query_rewrite": query_rewrite_node,
    "retrieve": retrieve_node,
    "compose": compose_node,
    "verify": verify_node,
    "repair": repair_node,
    "finalize": finalize_node,
    "stm_write": stm_write_node,
    "stm_summary": stm_summary_node,
    "persist_ltm": persist_ltm_node,
    # Runtime V2 nodes
    "resolve_query_reference": resolve_query_reference_node,
    "rewrite_query_for_retrieval": rewrite_query_for_retrieval_node,
    "ltm_recall": ltm_recall_node,
    "compose_memory_draft": compose_memory_draft_node,
    "verify_memory": verify_memory_node,
    "retrieve_phase1": retrieve_phase1_node,
    "rerank_phase1": rerank_phase1_node,
    "compose_with_context": compose_with_context_node,
    "strict_verify": strict_verify_node,
    "degrade_or_abstain": degrade_or_abstain_node,
    "retrieve_phase2": retrieve_phase2_node,
    "rerank_phase2": rerank_phase2_node,
    "set_repair_mode": set_repair_mode_node,
}


NODE_CONTRACTS: dict[str, NodeContract] = {
    "stm_read": NodeContract(
        name="stm_read",
        reads={"session_id"},
        writes={"short_term_memory", "recent_messages", "_stm_to_compress", "steps_log"},
    ),
    "intent": NodeContract(
        name="intent",
        reads={"query", "short_term_memory", "recent_messages"},
        writes={"intent", "steps_log"},
        llm_node=True,
    ),
    "planner": NodeContract(
        name="planner",
        reads={"query", "intent"},
        writes={
            "plan",
            "is_direct_path",
            "loop_count",
            "repair_hint",
            "rewritten_query",
            "context_pool",
            "long_term_memory",
            "steps_log",
        },
    ),
    "recall_ltm": NodeContract(
        name="recall_ltm",
        reads={"query"},
        writes={"long_term_memory", "steps_log"},
    ),
    "query_rewrite": NodeContract(
        name="query_rewrite",
        reads={"query", "long_term_memory"},
        writes={"rewritten_query", "steps_log"},
        llm_node=True,
    ),
    "retrieve": NodeContract(
        name="retrieve",
        reads={"query", "rewritten_query"},
        writes={"context_pool", "steps_log"},
    ),
    "compose": NodeContract(
        name="compose",
        reads={"query", "context_pool", "short_term_memory", "long_term_memory", "recent_messages", "repair_hint"},
        writes={"response", "citations", "steps_log"},
        llm_node=True,
    ),
    "verify": NodeContract(
        name="verify",
        reads={"response", "context_pool"},
        writes={"is_hallucination", "verify_score", "critique", "repair_hint", "steps_log"},
        llm_node=True,
    ),
    "repair": NodeContract(
        name="repair",
        reads={"critique", "plan"},
        writes={"repair_hint", "context_pool", "steps_log"},
    ),
    "finalize": NodeContract(
        name="finalize",
        reads={"response"},
        writes={"response"},
    ),
    "stm_write": NodeContract(
        name="stm_write",
        reads={"response", "session_id"},
        writes={"steps_log"},
    ),
    "stm_summary": NodeContract(
        name="stm_summary",
        reads={"session_id"},
        writes={"steps_log"},
    ),
    "persist_ltm": NodeContract(
        name="persist_ltm",
        reads={"query", "response"},
        writes={"steps_log"},
    ),
    # Runtime V2 contracts
    "resolve_query_reference": NodeContract(
        name="resolve_query_reference",
        reads={"query", "short_term_memory", "recent_messages"},
        writes={"resolved_query", "steps_log"},
        llm_node=True,
    ),
    "rewrite_query_for_retrieval": NodeContract(
        name="rewrite_query_for_retrieval",
        reads={"query", "resolved_query", "short_term_memory", "recent_messages", "long_term_memory"},
        writes={"retrieval_query", "steps_log"},
        llm_node=True,
    ),
    "ltm_recall": NodeContract(
        name="ltm_recall",
        reads={"query", "resolved_query"},
        writes={"long_term_memory", "ltm_hits_count", "steps_log"},
    ),
    "compose_memory_draft": NodeContract(
        name="compose_memory_draft",
        reads={"query", "short_term_memory", "recent_messages", "long_term_memory", "ltm_hits_count"},
        writes={"draft_answer", "draft_confidence", "used_memory_chunks", "steps_log"},
        llm_node=True,
    ),
    "verify_memory": NodeContract(
        name="verify_memory",
        reads={"query", "draft_answer", "draft_confidence", "used_memory_chunks", "ltm_hits_count"},
        writes={"memory_score", "memory_verdict", "memory_reason", "memory_risk_level", "steps_log"},
        llm_node=True,
    ),
    "retrieve_phase1": NodeContract(
        name="retrieve_phase1",
        reads={"query", "resolved_query", "retrieval_query"},
        writes={"phase1_candidates", "steps_log"},
    ),
    "rerank_phase1": NodeContract(
        name="rerank_phase1",
        reads={"query", "phase1_candidates"},
        writes={"phase1_candidates", "phase1_reranked", "context_pool", "context_source", "steps_log"},
    ),
    "compose_with_context": NodeContract(
        name="compose_with_context",
        reads={
            "query",
            "short_term_memory",
            "recent_messages",
            "long_term_memory",
            "context_pool",
            "repair_mode",
            "failure_type",
            "strict_reason",
            "previous_response",
        },
        writes={"response", "citations", "response_revision", "previous_response", "steps_log"},
        llm_node=True,
    ),
    "strict_verify": NodeContract(
        name="strict_verify",
        reads={"query", "response", "citations", "context_pool", "response_revision"},
        writes={"strict_score", "strict_verdict", "failure_type", "strict_reason", "verified_revision", "steps_log"},
        llm_node=True,
    ),
    "degrade_or_abstain": NodeContract(
        name="degrade_or_abstain",
        reads={"strict_reason", "failure_type"},
        writes={"response", "citations", "run_status", "steps_log"},
    ),
    "retrieve_phase2": NodeContract(
        name="retrieve_phase2",
        reads={"query", "resolved_query", "retrieval_query"},
        writes={"phase2_candidates", "steps_log"},
    ),
    "rerank_phase2": NodeContract(
        name="rerank_phase2",
        reads={"query", "phase2_candidates"},
        writes={"phase2_candidates", "phase2_reranked", "context_pool", "context_source", "steps_log"},
    ),
    "set_repair_mode": NodeContract(
        name="set_repair_mode",
        reads={"strict_verdict", "failure_type", "strict_reason", "repair_used"},
        writes={"repair_mode", "repair_used", "repair_reason", "steps_log"},
    ),
}


def _changed_keys(before_state: dict[str, Any], after_state: dict[str, Any]) -> set[str]:
    changed: set[str] = set()
    keys = set(before_state.keys()) | set(after_state.keys())
    for key in keys:
        before = before_state.get(key, None)
        after = after_state.get(key, None)
        if key not in before_state or key not in after_state or before != after:
            changed.add(key)
    return changed


def validate_node_contract(
    node_name: str,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    contracts: dict[str, NodeContract] | None = None,
) -> dict[str, Any]:
    contract_map = contracts or NODE_CONTRACTS
    contract = contract_map.get(node_name)
    if contract is None:
        return {
            "node": node_name,
            "enforced": False,
            "valid": True,
            "missing_reads": [],
            "unexpected_writes": [],
            "changed_keys": [],
        }

    missing_reads = sorted(k for k in contract.reads if k not in before_state)
    changed_keys = sorted(_changed_keys(before_state, after_state))
    unexpected_writes = sorted(k for k in changed_keys if k not in contract.writes)

    # strict=False 时只做观测，不拦截执行
    valid = (not missing_reads) and (not unexpected_writes or not contract.strict)

    return {
        "node": node_name,
        "enforced": bool(contract.strict),
        "valid": valid,
        "missing_reads": missing_reads,
        "unexpected_writes": unexpected_writes,
        "changed_keys": changed_keys,
    }
