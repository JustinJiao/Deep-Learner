import time
import re
from pathlib import Path

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
    top_k = int(AppConfig.RUNTIME_COMPOSE_CONTEXT_TOP_K)
    if top_k <= 0:
        return list(context_pool)
    return list(context_pool[:top_k])


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_heading(value: object) -> str:
    text = str(value or "").strip()
    while text.startswith("#"):
        text = text[1:]
    return text.strip()


def _extract_source_and_module(doc: dict) -> tuple[str, str]:
    metadata = doc.get("metadata", {}) or {}

    source_raw = (
        metadata.get("source")
        or doc.get("source")
        or doc.get("title")
        or doc.get("id")
        or "Unknown Document"
    )
    source_text = str(source_raw).strip()
    source_name = Path(source_text).name or source_text or "Unknown Document"

    module = (
        _normalize_heading(metadata.get("h2"))
        or _normalize_heading(metadata.get("h1"))
        or _normalize_heading(doc.get("module"))
        or "General"
    )
    return source_name, module


def _build_compose_prompt_context(context_pool: list[dict]) -> list[dict]:
    prompt_docs: list[dict] = []
    for doc in context_pool:
        source_name, module = _extract_source_and_module(doc)
        doc_id = str(doc.get("id", "")).strip() or f"{source_name}::{module}"
        prompt_docs.append(
            {
                "id": doc_id,
                "title": source_name,
                "source": source_name,
                "module": module,
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

_COMPANY_ALIAS_MAP: dict[str, str] = {
    "amazon": "Amazon",
    "microsoft": "Microsoft",
    "msft": "Microsoft",
    "alphabet": "Alphabet",
    "google": "Alphabet",
}

_MULTI_COMPANY_MARKERS = (
    "all three companies",
    "all three",
    "all companies",
    "these companies",
    "three companies",
)


def _query_terms(query: str) -> set[str]:
    tokens = _TOKEN_PATTERN.findall(str(query or "").lower())
    return {tok for tok in tokens if len(tok) >= 3}


def _companies_from_text(text: str) -> set[str]:
    lowered = str(text or "").lower()
    found: set[str] = set()
    for alias, company in _COMPANY_ALIAS_MAP.items():
        if alias in lowered:
            found.add(company)
    return found


def _is_generic_multi_company_query(query: str) -> bool:
    lowered = str(query or "").lower()
    return any(marker in lowered for marker in _MULTI_COMPANY_MARKERS)


def _required_companies_for_query(query: str, context_pool: list[dict]) -> set[str]:
    project_default = {"Amazon", "Alphabet", "Microsoft"}
    query_companies = _companies_from_text(query)
    if len(query_companies) >= 2:
        return query_companies
    if not _is_generic_multi_company_query(query):
        return set()

    inferred: set[str] = set()
    for doc in context_pool:
        metadata = doc.get("metadata", {}) or {}
        source_text = " ".join(
            [
                str(metadata.get("source", "") or ""),
                str(doc.get("source", "") or ""),
                str(doc.get("title", "") or ""),
                str(doc.get("id", "") or ""),
            ]
        )
        inferred.update(_companies_from_text(source_text))
    if len(inferred) >= 2:
        return inferred
    return project_default


def _companies_from_citations(citations: list[dict]) -> set[str]:
    found: set[str] = set()
    for c in citations:
        text = " ".join(
            [
                str(c.get("id", "") or ""),
                str(c.get("title", "") or ""),
            ]
        )
        found.update(_companies_from_text(text))
    return found


def _missing_company_coverage(
    query: str,
    context_pool: list[dict],
    citations: list[dict],
) -> set[str]:
    required = _required_companies_for_query(query, context_pool)
    if len(required) < 2:
        return set()
    cited = _companies_from_citations(citations)
    return required - cited


def _single_company_target(query: str) -> str | None:
    companies = _companies_from_text(query)
    if len(companies) != 1:
        return None
    return next(iter(companies))


def _out_of_scope_companies_for_single_target(
    query: str,
    response: str,
    citations: list[dict],
) -> set[str]:
    target = _single_company_target(query)
    if not target:
        return set()
    mentioned = _companies_from_text(response) | _companies_from_citations(citations)
    extras = {c for c in mentioned if c != target}
    return extras


def _companies_from_doc_source(doc: dict) -> set[str]:
    metadata = doc.get("metadata", {}) or {}
    source_text = " ".join(
        [
            str(metadata.get("source", "") or ""),
            str(doc.get("source", "") or ""),
            str(doc.get("title", "") or ""),
            str(doc.get("id", "") or ""),
        ]
    )
    return _companies_from_text(source_text)


def _ensure_compose_company_coverage(
    query: str,
    compose_context_pool: list[dict],
    full_context_pool: list[dict],
) -> list[dict]:
    required = _required_companies_for_query(query, full_context_pool)
    if len(required) < 2:
        return compose_context_pool

    composed = list(compose_context_pool)
    existing_ids = {str(d.get("id", "")).strip() for d in composed}
    covered: set[str] = set()
    for doc in composed:
        covered.update(_companies_from_doc_source(doc))

    if required.issubset(covered):
        return composed

    for doc in full_context_pool:
        doc_id = str(doc.get("id", "")).strip()
        if not doc_id or doc_id in existing_ids:
            continue
        doc_companies = _companies_from_doc_source(doc)
        if not doc_companies:
            continue
        if not (doc_companies & (required - covered)):
            continue
        composed.append(doc)
        existing_ids.add(doc_id)
        covered.update(doc_companies)
        if required.issubset(covered):
            break

    return composed


def _build_partial_multi_company_response(
    response: str,
    citations: list[dict],
    missing_companies: set[str],
) -> str:
    missing_text = ", ".join(sorted(missing_companies))
    covered = sorted(_companies_from_citations(citations))

    base = str(response or "").strip()
    if _is_uncertain_response(base) or not base:
        if covered:
            base = (
                "Based on the currently retrieved evidence, this is a partial answer "
                f"covering: {', '.join(covered)}."
            )
        else:
            base = "The current retrieved evidence is insufficient for a complete cross-company answer."

    if re.search(r"missing\s+explicit\s+evidence\s+for\s*:", base, flags=re.IGNORECASE):
        return base

    return f"{base}\n\nMissing explicit evidence for: {missing_text}."


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
    if not bool(AppConfig.RUNTIME_REPAIR_EXTRACTIVE_FALLBACK):
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
    compose_context_pool = _ensure_compose_company_coverage(
        query=str(state.get("query", "")),
        compose_context_pool=compose_context_pool,
        full_context_pool=full_context_pool,
    )
    prompt_context_pool = _build_compose_prompt_context(compose_context_pool)

    prompt_state = _build_prompt_state(state, prompt_context_pool)
    out = run_prompt(ComposeWithContextPrompt, prompt_state)

    response = str(out.get("response", "")).strip()
    citations = out.get("citations", []) or []
    forced_retry = False
    multi_company_retry = False
    single_company_retry = False
    repair_extractive_applied = False

    # 若模型在已有证据下仍输出“不确定”，触发一次强制抽取重试。
    if (
        bool(AppConfig.RUNTIME_FORCE_ANSWER_ON_EVIDENCE)
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

    # 单公司题：若回答/引用扩展到其它公司，触发一次范围收敛重试。
    out_of_scope_companies = _out_of_scope_companies_for_single_target(
        query=str(state.get("query", "")),
        response=response,
        citations=citations,
    )
    if out_of_scope_companies:
        single_company_retry = True
        target = _single_company_target(str(state.get("query", ""))) or "target company"
        retry_state = _build_prompt_state(state, prompt_context_pool)
        retry_state["repair_mode"] = True
        retry_state["strict_reason"] = (
            "Scope error for single-company question. "
            f"The user asked only about {target}. "
            f"Remove discussion of non-target companies: {', '.join(sorted(out_of_scope_companies))}. "
            "Answer only for the target company with target-company citations."
        )
        retry_state["previous_response"] = response
        retry_out = run_prompt(ComposeWithContextPrompt, retry_state)
        retry_response = str(retry_out.get("response", "")).strip()
        retry_citations = retry_out.get("citations", []) or []
        if retry_response:
            response = retry_response
            citations = retry_citations

    # 多公司题：若引用未覆盖所有公司，强制一次补证重试。
    missing_company_coverage = _missing_company_coverage(
        query=str(state.get("query", "")),
        context_pool=full_context_pool,
        citations=citations,
    )
    if missing_company_coverage:
        multi_company_retry = True
        retry_state = _build_prompt_state(state, prompt_context_pool)
        retry_state["repair_mode"] = True
        retry_state["strict_reason"] = (
            "Multi-company evidence coverage is incomplete. "
            f"Missing companies in citations: {', '.join(sorted(missing_company_coverage))}. "
            "Revise the answer and include at least one citation per company/source. "
            "Use as many citations as needed."
        )
        retry_state["previous_response"] = response
        retry_out = run_prompt(ComposeWithContextPrompt, retry_state)
        retry_response = str(retry_out.get("response", "")).strip()
        retry_citations = retry_out.get("citations", []) or []
        if retry_response:
            response = retry_response
            citations = retry_citations

        missing_after_retry = _missing_company_coverage(
            query=str(state.get("query", "")),
            context_pool=full_context_pool,
            citations=citations,
        )
        if missing_after_retry:
            response = _build_partial_multi_company_response(
                response=response,
                citations=citations,
                missing_companies=missing_after_retry,
            )

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
                    "multi_company_retry": multi_company_retry,
                    "single_company_retry": single_company_retry,
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
