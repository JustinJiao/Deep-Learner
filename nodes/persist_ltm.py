# nodes/persist_ltm.py
import time

from config.settings import AppConfig
from core.state import AgentState, StepLog
from memory.ltm import LTM
from core.llm_call import run_prompt
from llm.prompts.ltm_fact_extract import LTMFactExtractPrompt
from nodes.log_utils import clip_text


def _normalized_ltm_importance_threshold() -> float:
    threshold = float(AppConfig.LTM_IMPORTANCE_THRESHOLD)
    return max(0.0, min(1.0, threshold))


def _to_float_score(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def persist_ltm_node(state: AgentState) -> AgentState:
    if not bool(AppConfig.LTM_WRITE_ENABLED):
        state.setdefault("steps_log", []).append(
            StepLog(
                node="persist_ltm",
                info={
                    "state": {
                        "persist_skipped": True,
                        "reason": "LTM_WRITE_ENABLED=false",
                        "query_preview": clip_text(state.get("query", ""), 160),
                    },
                },
                timestamp=time.time(),
            )
        )
        return state

    # 没有 response 就不抽取，避免污染
    if not state.get("response"):
        state.setdefault("steps_log", []).append(
            StepLog(
                node="persist_ltm",
                info={
                    "state": {
                        "persist_skipped": True,
                        "reason": "empty_response",
                        "query_preview": clip_text(state.get("query", ""), 160),
                    },
                },
                timestamp=time.time(),
            )
        )
        return state

    out = run_prompt(LTMFactExtractPrompt, state)
    raw_candidates = out.get("fact_candidates", []) or []
    threshold = _normalized_ltm_importance_threshold()

    candidates = []
    for c in raw_candidates:
        score = _to_float_score(c.get("score"))
        if score < threshold:
            continue
        if not c.get("key") or not c.get("content"):
            continue
        normalized = dict(c)
        normalized["score"] = score
        candidates.append(normalized)

    ltm = LTM()
    stored = ltm.upsert(candidates)

    state.setdefault("steps_log", []).append(
        StepLog(
            node="persist_ltm",
            info={
                "state": {
                    "query_preview": clip_text(state.get("query", ""), 160),
                    "response_preview": clip_text(state.get("response", ""), 180),
                },
                "llm_output": {
                    "importance_threshold": threshold,
                    "fact_candidates_raw_count": len(raw_candidates),
                    "fact_candidates_kept_count": len(candidates),
                    "fact_candidates_preview": [
                        {
                            "key": c.get("key"),
                            "type": c.get("type"),
                            "score": c.get("score"),
                            "content_preview": clip_text(c.get("content", ""), 120),
                        }
                        for c in candidates[:4]
                    ],
                },
                "memory": {
                    "stored_count": stored,
                },
            },
            timestamp=time.time(),
        )
    )
    return state
