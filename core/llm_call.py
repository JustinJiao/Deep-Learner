# core/llm_call.py

import json
import re
from typing import Any, Dict, Type

from llm.prompts.base import PromptContract, PromptContractError
from config.factory import ResourceFactory


def _resolve_prompt_task(prompt_cls: Type[PromptContract]) -> str:
    name = prompt_cls.__name__
    if name in {"ComposePrompt", "ComposeWithContextPrompt", "ComposeEvidenceTablePrompt"}:
        return "compose"
    if name in {"VerifyPrompt", "StrictVerifyPrompt"}:
        return "verify"
    if name in {"RewritePrompt", "RewriteRetrievalQueryPrompt", "ResolveQueryReferencePrompt"}:
        return "rewrite"
    if name in {
        "ComposeMemoryDraftPrompt",
        "VerifyMemoryPrompt",
        "STMCompressPrompt",
        "LTMFactExtractPrompt",
    }:
        return "memory"
    return "default"


def _resolve_prompt_node(prompt_cls: Type[PromptContract]) -> str:
    name = prompt_cls.__name__
    mapping = {
        "IntentPrompt": "intent",
        "RewritePrompt": "query_rewrite",
        "ComposePrompt": "compose",
        "VerifyPrompt": "verify",
        "ResolveQueryReferencePrompt": "resolve_query_reference",
        "RewriteRetrievalQueryPrompt": "rewrite_query_for_retrieval",
        "ComposeMemoryDraftPrompt": "compose_memory_draft",
        "ComposeEvidenceTablePrompt": "compose_with_context",
        "VerifyMemoryPrompt": "verify_memory",
        "ComposeWithContextPrompt": "compose_with_context",
        "StrictVerifyPrompt": "strict_verify",
        "STMCompressPrompt": "stm_summary",
        "LTMFactExtractPrompt": "persist_ltm",
    }
    return mapping.get(name, "unknown")


def _try_parse_json_dict(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _repair_json_candidate(text: str) -> str:
    """
    Best-effort repair for truncated/dirty JSON object text:
    - keep from first '{'
    - remove unmatched closing brackets
    - auto-close open quotes/brackets
    - drop trailing commas before } or ]
    """
    raw = str(text or "").strip()
    if not raw:
        return ""

    start = raw.find("{")
    if start == -1:
        return ""
    raw = raw[start:]

    out_chars: list[str] = []
    closing_stack: list[str] = []
    in_string = False
    escaped = False

    for ch in raw:
        if in_string:
            out_chars.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            out_chars.append(ch)
            continue

        if ch == "{":
            closing_stack.append("}")
            out_chars.append(ch)
            continue

        if ch == "[":
            closing_stack.append("]")
            out_chars.append(ch)
            continue

        if ch in "}]":
            if closing_stack and ch == closing_stack[-1]:
                closing_stack.pop()
                out_chars.append(ch)
            # unmatched closing token -> drop
            continue

        out_chars.append(ch)

    repaired = "".join(out_chars).strip()
    if not repaired:
        return ""

    if in_string:
        repaired += '"'

    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

    if closing_stack:
        repaired += "".join(reversed(closing_stack))

    return repaired


def _try_extract_json_object(raw: str) -> tuple[dict[str, Any] | None, str]:
    text = str(raw or "").strip()
    if not text:
        return None, ""

    candidates: list[str] = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    if start != -1:
        candidates.append(text[start:])

    seen: set[str] = set()
    for cand in candidates:
        normalized = str(cand or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        parsed = _try_parse_json_dict(normalized)
        if parsed is not None:
            return parsed, "json_recovered"

        repaired = _repair_json_candidate(normalized)
        if repaired and repaired != normalized:
            parsed = _try_parse_json_dict(repaired)
            if parsed is not None:
                return parsed, "json_repaired"

    return None, ""


def run_prompt(
    prompt_cls: Type[PromptContract],
    state: Dict[str, Any],
) -> Dict[str, Any]:

    prompt = prompt_cls()
    prompt.validate_reads(state)

    system_prompt = prompt.build_system_prompt()
    user_prompt = prompt.build_user_prompt(state)

    llm = ResourceFactory.get_llm_service()
    task = _resolve_prompt_task(prompt_cls)
    node_name = _resolve_prompt_node(prompt_cls)

    trace_event: dict[str, Any] = {
        "prompt": prompt_cls.__name__,
        "node": node_name,
        "task": task,
        "provider": "",
        "model": "",
        "latency_ms": 0.0,
        "parse_mode": "unknown",
        "error": "",
        "writes": list(prompt.WRITES),
    }
    output: dict[str, Any] | None = None

    try:
        payload = llm.chat_completion_with_meta(
            prompt=user_prompt,
            system_prompt=system_prompt,
            task=task,
        )
        raw = str(payload.get("content", ""))
        trace_event["provider"] = str(payload.get("provider", ""))
        trace_event["model"] = str(payload.get("model", ""))
        trace_event["latency_ms"] = float(payload.get("latency_ms", 0.0) or 0.0)

        # ---- 如果是生成类 prompt（Compose） ----
        if prompt.WRITES == ["response"]:
            trace_event["parse_mode"] = "plain_text"
            output = {"response": raw.strip()}
            return output

        # ---- 其余必须 JSON ----
        try:
            output = json.loads(raw)
            trace_event["parse_mode"] = "json"
        except json.JSONDecodeError:
            recovered, recovered_mode = _try_extract_json_object(raw)
            if recovered is not None:
                output = recovered
                trace_event["parse_mode"] = recovered_mode or "json_recovered"
            elif set(prompt.WRITES) == {"response", "citations"}:
                # 某些模型偶发不输出 JSON，兜底为文本响应，避免整链路报错中断。
                output = {"response": str(raw).strip(), "citations": []}
                trace_event["parse_mode"] = "compose_text_fallback"
            else:
                trace_event["parse_mode"] = "json_error"
                raise PromptContractError(
                    f"[PromptContract] {prompt_cls.__name__} must output valid JSON.\n"
                    f"Got: {raw[:300]}"
                )

        if not isinstance(output, dict):
            raise PromptContractError(
                f"[PromptContract] {prompt_cls.__name__} output must be a JSON object."
            )

        missing = [k for k in prompt.WRITES if k not in output]
        if missing:
            raise PromptContractError(
                f"[PromptContract] Missing WRITES fields for {prompt_cls.__name__}: {missing}"
            )

        return output
    except Exception as e:
        trace_event["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        if output is not None:
            try:
                trace_event["output_chars"] = len(json.dumps(output, ensure_ascii=False))
            except Exception:
                trace_event["output_chars"] = 0
        else:
            trace_event["output_chars"] = 0

        state.setdefault("llm_trace", []).append(trace_event)
