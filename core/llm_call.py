# core/llm_call.py

import json
from typing import Type, Dict, Any
from llm.prompts.base import PromptContract, PromptContractError
from config.factory import ResourceFactory

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
        system_prompt=system_prompt
    )

    # ---- 如果是生成类 prompt（Compose） ----
    if prompt.WRITES == ["response"]:
        return {"response": raw.strip()}

    # ---- 其余必须 JSON ----
    try:
        output = json.loads(raw)
    except json.JSONDecodeError:
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
