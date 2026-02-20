# nodes/recall_ltm.py
import time

from core.state import AgentState, StepLog
from memory.ltm import LTM
from nodes.log_utils import clip_text


def recall_ltm_node(state: AgentState) -> AgentState:
    ltm = LTM()
    memories = ltm.recall(state["query"])

    state["long_term_memory"] = " | ".join(memories) if memories else "无相关长期记忆"

    state.setdefault("steps_log", []).append(
        StepLog(
            node="recall_ltm",
            info={
                "state": {
                    "query_preview": clip_text(state.get("query", ""), 180),
                },
                "memory": {
                    "memory_count": len(memories),
                    "memories_preview": [clip_text(m, 160) for m in memories[:4]],
                },
                "llm_input": {
                    "long_term_memory_preview": clip_text(state.get("long_term_memory", ""), 180),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
