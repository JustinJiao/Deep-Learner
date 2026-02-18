# tests/prompts/_helpers.py
import pytest

def assert_validate_reads_ok(prompt_cls, state):
    """
    确认 READS 满足时不会抛异常
    """
    try:
        prompt_cls.validate_reads(state)
    except Exception as e:
        pytest.fail(f"validate_reads raised unexpectedly: {e}")


def assert_validate_reads_fail(prompt_cls, state):
    """
    确认 READS 不满足时一定抛异常
    """
    with pytest.raises(ValueError):
        prompt_cls.validate_reads(state)


def assert_build_prompt_ok(prompt_cls, state):
    """
    build_user_prompt 必须能正常运行并返回 str
    """
    prompt = prompt_cls.build_user_prompt(state)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
