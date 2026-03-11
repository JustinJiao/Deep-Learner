import time
import re

from core.llm_call import run_prompt
from core.state import AgentState, StepLog
from llm.prompts.resolve_query_reference import ResolveQueryReferencePrompt
from llm.prompts.base import PromptContractError
from nodes.log_utils import clip_text, preview_messages

_LATEST_FISCAL_YEAR_MARKERS = (
    "most recent fiscal year",
    "latest fiscal year",
    "recent fiscal year",
    "most recent year",
    "latest year",
)

_MULTI_COMPANY_MARKERS = (
    "all three companies",
    "all companies",
    "these companies",
    "all three",
    "three companies",
)

_COMPANY_ALIASES: dict[str, str] = {
    "amazon": "Amazon",
    "alphabet": "Alphabet",
    "google": "Alphabet",
    "microsoft": "Microsoft",
    "msft": "Microsoft",
}

_PROJECT_DEFAULT_COMPANIES = ("Amazon", "Alphabet", "Microsoft")


def _companies_from_text(text: str) -> list[str]:
    lowered = str(text or "").lower()
    found: list[str] = []
    seen: set[str] = set()
    for alias, name in _COMPANY_ALIASES.items():
        if alias in lowered and name not in seen:
            found.append(name)
            seen.add(name)
    return found


def _extract_latest_year_from_memory(long_term_memory: str) -> int | None:
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", str(long_term_memory or ""))]
    if not years:
        return None
    return max(years)


def _contains_latest_fiscal_marker(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in _LATEST_FISCAL_YEAR_MARKERS)


def _contains_multi_company_marker(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in _MULTI_COMPANY_MARKERS)


def _contains_explicit_year(text: str) -> bool:
    return bool(re.search(r"\b20\d{2}\b", str(text or "")))


def _apply_background_normalization(
    original_query: str,
    resolved_query: str,
    long_term_memory: str,
) -> tuple[str, list[str]]:
    text = str(resolved_query or original_query or "").strip()
    tags: list[str] = []

    latest_year = _extract_latest_year_from_memory(long_term_memory)
    if latest_year and _contains_latest_fiscal_marker(text) and not _contains_explicit_year(text):
        text = re.sub(
            r"(most recent fiscal year|latest fiscal year|recent fiscal year|most recent year|latest year)",
            f"fiscal year {latest_year}",
            text,
            flags=re.IGNORECASE,
        )
        tags.append("latest_fiscal_year_normalized")

    companies_in_query = _companies_from_text(text)
    if _contains_multi_company_marker(text) and len(companies_in_query) < 2:
        mem_companies = _companies_from_text(long_term_memory)
        if len(mem_companies) < 2:
            mem_companies = list(_PROJECT_DEFAULT_COMPANIES)

        joined = ", ".join(mem_companies)
        text = re.sub(
            r"(all three companies|all companies|these companies|all three|three companies)",
            f"{joined}",
            text,
            flags=re.IGNORECASE,
        )
        tags.append("multi_company_expanded_from_memory")

    return text, tags


def resolve_query_reference_node(state: AgentState) -> AgentState:
    original_query = str(state.get("query", "")).strip()
    long_term_memory = str(state.get("long_term_memory", "") or "")
    llm_error = ""
    used_fallback = False
    memory_normalization_tags: list[str] = []
    out: dict = {}

    try:
        out = run_prompt(ResolveQueryReferencePrompt, state)
    except PromptContractError as e:
        # JSON 解析失败等格式问题时，不中断主流程，直接回退到原 query。
        llm_error = f"{type(e).__name__}: {e}"
        used_fallback = True

    resolved_query = str(out.get("resolved_query", "")).strip()

    if not resolved_query:
        resolved_query = original_query
        used_fallback = True

    resolved_query, memory_normalization_tags = _apply_background_normalization(
        original_query=original_query,
        resolved_query=resolved_query,
        long_term_memory=long_term_memory,
    )

    state["resolved_query"] = resolved_query

    state.setdefault("steps_log", []).append(
        StepLog(
            node="resolve_query_reference",
            info={
                "state": {
                    "query_preview": clip_text(original_query, 180),
                },
                "llm_input": {
                    "short_term_memory_preview": clip_text(state.get("short_term_memory", ""), 160),
                    "recent_messages_preview": preview_messages(state.get("recent_messages", [])),
                    "long_term_memory_preview": clip_text(long_term_memory, 200),
                },
                "llm_output": {
                    "resolved_query_preview": clip_text(resolved_query, 180),
                    "used_fallback": used_fallback,
                    "memory_normalization_tags": memory_normalization_tags,
                    "llm_error_preview": clip_text(llm_error, 220),
                },
            },
            timestamp=time.time(),
        )
    )
    return state
