import time
import re

from config.settings import AppConfig
from core.llm_call import run_prompt
from core.state import AgentState, StepLog
from llm.prompts.compose_with_context import ComposeWithContextPrompt
from nodes.log_utils import clip_text, preview_citations, preview_docs, preview_messages


_UNCERTAIN_MARKERS = (
    "不确定",
    "无法确定",
    "证据不足",
    "无法给出可验证的结论",
    "uncertain",
    "cannot determine",
    "insufficient evidence",
    "not sure",
)


def _is_uncertain_response(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    lowered = raw.lower()
    return any(marker in raw or marker in lowered for marker in _UNCERTAIN_MARKERS)


def _pick_compose_context(context_pool: list[dict]) -> list[dict]:
    top_k = int(getattr(AppConfig, "RUNTIME_COMPOSE_CONTEXT_TOP_K", 8))
    if top_k <= 0:
        return list(context_pool)
    return list(context_pool[:top_k])


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_compose_prompt_context(context_pool: list[dict]) -> list[dict]:
    prompt_docs: list[dict] = []
    for doc in context_pool:
        prompt_docs.append(
            {
                "id": str(doc.get("id", "")).strip(),
                "title": str(doc.get("title", "")).strip(),
                "score": _safe_float(doc.get("score"), 0.0),
                "content": str(doc.get("content", "") or ""),
            }
        )
    return prompt_docs


def _context_chars(context_pool: list[dict]) -> int:
    chars = 0
    for doc in context_pool:
        chars += len(str(doc.get("content", "") or ""))
    return chars


def _build_prompt_state(state: AgentState, context_pool: list[dict]) -> AgentState:
    prompt_state: AgentState = dict(state)
    prompt_state["context_pool"] = context_pool
    return prompt_state


def _fallback_citation(doc: dict, quote_override: str | None = None) -> dict | None:
    doc_id = str(doc.get("id", "")).strip()
    if not doc_id:
        return None

    title = str(doc.get("title", "") or "Untitled")
    score = float(doc.get("score", 0.0))
    content = str(doc.get("content", "") or "").strip()
    quote = str(quote_override or content).replace("\n", " ").strip()
    if len(quote) > 180:
        quote = quote[:180]
    if not quote:
        quote = title
    return {
        "id": doc_id,
        "title": title,
        "score": score,
        "quote": quote,
    }


def _ensure_non_empty_citations(response: str, citations: list[dict], context_pool: list[dict]) -> list[dict]:
    if citations:
        return citations
    if _is_uncertain_response(response):
        return []
    if not context_pool:
        return []
    fallback = _fallback_citation(context_pool[0])
    return [fallback] if fallback else []


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _query_terms(query: str) -> set[str]:
    tokens = _TOKEN_PATTERN.findall(str(query or "").lower())
    return {tok for tok in tokens if len(tok) >= 3}


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", str(text or ""))
    return [p.strip() for p in parts if p and p.strip()]


def _best_extractive_snippet(query: str, content: str) -> str:
    sentences = _split_sentences(content)
    if not sentences:
        return str(content or "").strip()[:180]

    q_terms = _query_terms(query)
    if not q_terms:
        return sentences[0][:180]

    best_sentence = sentences[0]
    best_score = -1
    for sent in sentences:
        s_terms = set(_TOKEN_PATTERN.findall(sent.lower()))
        overlap = len(q_terms & s_terms)
        score = overlap * 100 + min(len(sent), 200)
        if score > best_score:
            best_score = score
            best_sentence = sent
    return best_sentence[:180]


def _select_repair_anchor_doc(context_pool: list[dict], citations: list[dict]) -> dict | None:
    if not context_pool:
        return None

    citation_id = ""
    if citations:
        citation_id = str((citations[0] or {}).get("id", "")).strip()
    if citation_id:
        for doc in context_pool:
            if str(doc.get("id", "")).strip() == citation_id:
                return doc
    return context_pool[0]


def _should_apply_repair_extractive_fallback(
    state: AgentState,
    response: str,
    citations: list[dict],
    context_pool: list[dict],
) -> bool:
    if not bool(state.get("repair_mode", False)):
        return False
    if not bool(getattr(AppConfig, "RUNTIME_REPAIR_EXTRACTIVE_FALLBACK", True)):
        return False
    if not context_pool:
        return False

    failure_type = str(state.get("failure_type", "")).strip().upper()
    if failure_type in {"LOGICAL_ERROR", "CITATION_MISMATCH", "FORMAT_ERROR"}:
        return True

    return _is_uncertain_response(response) or (not citations)


def compose_with_context_node(state: AgentState) -> AgentState:
    state.setdefault("repair_mode", False)
    state.setdefault("strict_reason", "")
    state.setdefault("previous_response", state.get("response", ""))

    full_context_pool = state.get("context_pool", []) or []
    compose_context_pool = _pick_compose_context(full_context_pool)
    prompt_context_pool = _build_compose_prompt_context(compose_context_pool)

    prompt_state = _build_prompt_state(state, prompt_context_pool)
    out = run_prompt(ComposeWithContextPrompt, prompt_state)

    response = str(out.get("response", "")).strip()
    citations = out.get("citations", []) or []
    forced_retry = False
    repair_extractive_applied = False

    # 若模型在已有证据下仍输出“不确定”，触发一次强制抽取重试。
    if (
        bool(getattr(AppConfig, "RUNTIME_FORCE_ANSWER_ON_EVIDENCE", True))
        and compose_context_pool
        and _is_uncertain_response(response)
    ):
        forced_retry = True
        retry_state = _build_prompt_state(state, prompt_context_pool)
        retry_state["repair_mode"] = True
        retry_state["strict_reason"] = (
            "The previous answer was uncertain. Evidence exists in context_pool. "
            "Extract the best-supported direct answer and provide at least one citation."
        )
        retry_state["previous_response"] = response
        retry_out = run_prompt(ComposeWithContextPrompt, retry_state)
        retry_response = str(retry_out.get("response", "")).strip()
        retry_citations = retry_out.get("citations", []) or []
        if retry_response:
            response = retry_response
            citations = retry_citations

    # repair 阶段优先转为抽取式回答，避免再次发生逻辑扩写错误。
    if _should_apply_repair_extractive_fallback(
        state=state,
        response=response,
        citations=citations,
        context_pool=compose_context_pool,
    ):
        anchor_doc = _select_repair_anchor_doc(compose_context_pool, citations)
        if anchor_doc:
            snippet = _best_extractive_snippet(
                query=str(state.get("query", "")),
                content=str(anchor_doc.get("content", "")),
            )
            fallback_citation = _fallback_citation(anchor_doc, quote_override=snippet)
            if fallback_citation:
                response = snippet or response
                citations = [fallback_citation]
                repair_extractive_applied = True

    state["previous_response"] = state.get("response", "")
    state["response"] = response
    state["citations"] = _ensure_non_empty_citations(response, citations, compose_context_pool)
    state["response_revision"] = state.get("response_revision", 0) + 1

    state.setdefault("steps_log", []).append(
        StepLog(
            node="compose_with_context",
            info={
                "state": {
                    "query_preview": clip_text(state.get("query", ""), 180),
                    "repair_mode": bool(state.get("repair_mode", False)),
                    "context_pool_count": len(full_context_pool),
                    "compose_context_count": len(compose_context_pool),
                    "compose_context_chars": _context_chars(compose_context_pool),
                    "compose_prompt_context_chars": _context_chars(prompt_context_pool),
                    "forced_retry": forced_retry,
                    "repair_extractive_applied": repair_extractive_applied,
                    "response_revision": state.get("response_revision", 0),
                },
                "llm_input": {
                    "short_term_memory_preview": clip_text(state.get("short_term_memory", ""), 160),
                    "recent_messages_preview": preview_messages(state.get("recent_messages", [])),
                    "long_term_memory_preview": clip_text(state.get("long_term_memory", ""), 160),
                    "strict_reason_preview": clip_text(state.get("strict_reason", ""), 160),
                    "context_pool_preview": preview_docs(prompt_context_pool),
                },
                "llm_output": {
                    "response_preview": clip_text(state.get("response", ""), 220),
                    "citations_preview": preview_citations(state.get("citations", [])),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
