# nodes/persist_ltm.py
import time

from core.state import AgentState, StepLog
from memory.ltm import LTM
from core.llm_call import run_prompt
from llm.prompts.ltm_fact_extract import LTMFactExtractPrompt
from nodes.log_utils import clip_text


def persist_ltm_node(state: AgentState) -> AgentState:
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
    candidates = out.get("fact_candidates", []) or []

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
