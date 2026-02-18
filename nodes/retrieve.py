# nodes/retrieve.py
import time
from core.state import AgentState, StepLog
from tools.retrieve_tool.pipeline import RetrievalPipeline


def retrieve_node(state: AgentState) -> AgentState:
    pipeline = RetrievalPipeline()
    query = state.get("rewritten_query", state["query"])

    results = pipeline.run(query)

    state["context_pool"] = [
        {
            "id": r.id,
            "content": r.content,
            "metadata": r.metadata,
            "score": r.score,
            "source_type": r.source_type,
        }
        for r in results
    ]

    state.setdefault("steps_log", []).append(
        StepLog(
            node="retrieve",
            info=f"docs={len(results)}",
            timestamp=time.time(),
        )
    )
    return state
