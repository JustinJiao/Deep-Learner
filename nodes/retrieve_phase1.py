import time

from core.state import AgentState, StepLog
from nodes.log_utils import clip_text, preview_docs
from tools.retrieve_tool.vector import VectorRetriever

_VECTOR_RETRIEVER: VectorRetriever | None = None


def _get_retriever() -> VectorRetriever:
    global _VECTOR_RETRIEVER
    if _VECTOR_RETRIEVER is None:
        _VECTOR_RETRIEVER = VectorRetriever()
    return _VECTOR_RETRIEVER


def retrieve_phase1_node(state: AgentState) -> AgentState:
    original_query = str(state.get("query", "")).strip()
    retrieval_query = str(
        state.get("retrieval_query") or state.get("resolved_query") or original_query
    ).strip()
    results = _get_retriever().search(retrieval_query, top_k=30)

    phase1_candidates: list[dict] = []
    for r in results:
        title = None
        if isinstance(r.metadata, dict):
            title = r.metadata.get("title")
        phase1_candidates.append(
            {
                "id": r.id,
                "title": title or r.source_type or "Untitled",
                "content": r.content,
                "score": float(r.score),
                "metadata": r.metadata,
                "source_type": r.source_type,
            }
        )

    state["phase1_candidates"] = phase1_candidates

    state.setdefault("steps_log", []).append(
        StepLog(
            node="retrieve_phase1",
            info={
                "state": {
                    "query_preview": clip_text(original_query, 180),
                    "retrieval_query_preview": clip_text(retrieval_query, 180),
                    "used_rewritten_query": retrieval_query != original_query,
                },
                "memory": {
                    "top_k": 30,
                    "candidate_count": len(phase1_candidates),
                    "candidate_preview": preview_docs(phase1_candidates),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
