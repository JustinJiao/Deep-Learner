# tests/prompts/_helpers.py
import pytest

from llm.prompts.base import PromptContractError


def _build_prompt(prompt_cls):
    return prompt_cls()


def assert_validate_reads_ok(prompt_cls, state):
    """
    确认 READS 满足时不会抛异常
    """
    prompt = _build_prompt(prompt_cls)
    try:
        prompt.validate_reads(state)
    except Exception as e:
        pytest.fail(f"validate_reads raised unexpectedly: {e}")


def assert_validate_reads_fail(prompt_cls, state):
    """
    确认 READS 不满足时一定抛异常
    """
    prompt = _build_prompt(prompt_cls)
    with pytest.raises(PromptContractError):
        prompt.validate_reads(state)


def assert_build_prompt_ok(prompt_cls, state):
    """
    build_user_prompt 必须能正常运行并返回 str
    """
    prompt = _build_prompt(prompt_cls)
    rendered = prompt.build_user_prompt(state)
    assert isinstance(rendered, str)
    assert len(rendered) > 0
