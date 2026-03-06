# core/llm_call.py

import json
import re
from typing import Type, Dict, Any
from llm.prompts.base import PromptContract, PromptContractError
from config.factory import ResourceFactory


def _resolve_prompt_task(prompt_cls: Type[PromptContract]) -> str:
    name = prompt_cls.__name__
    if name in {"ComposePrompt", "ComposeWithContextPrompt"}:
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


def _try_extract_json_object(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    if not text:
        return None

    candidates: list[str] = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def run_prompt(
    prompt_cls: Type[PromptContract],
    state: Dict[str, Any],
) -> Dict[str, Any]:

    prompt = prompt_cls()
    prompt.validate_reads(state)

    system_prompt = prompt.build_system_prompt()
    user_prompt = prompt.build_user_prompt(state)

    llm = ResourceFactory.get_llm_service()

    raw = llm.chat_completion(
        prompt=user_prompt,
        system_prompt=system_prompt,
        task=_resolve_prompt_task(prompt_cls),
    )

    # ---- 如果是生成类 prompt（Compose） ----
    if prompt.WRITES == ["response"]:
        return {"response": raw.strip()}

    # ---- 其余必须 JSON ----
    try:
        output = json.loads(raw)
    except json.JSONDecodeError:
        recovered = _try_extract_json_object(raw)
        if recovered is not None:
            output = recovered
        elif set(prompt.WRITES) == {"response", "citations"}:
            # 某些模型偶发不输出 JSON，兜底为文本响应，避免整链路报错中断。
            output = {"response": str(raw).strip(), "citations": []}
        else:
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
