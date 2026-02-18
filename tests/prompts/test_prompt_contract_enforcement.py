import pytest
from llm.prompts.base import PromptContract


class BadPrompt(PromptContract):
    READS = ["a"]
    WRITES = ["b"]
    SYSTEM = "x"

    @staticmethod
    def build_user_prompt(state):
        return "test"


def test_missing_reads():
    with pytest.raises(ValueError):
        BadPrompt.validate_reads({})


def test_missing_writes():
    with pytest.raises(ValueError):
        BadPrompt.validate_writes({"a": 1})
