# llm/prompts/ltm_fact_extract.py

from llm.prompts.base import PromptContract


class LTMFactExtractPrompt(PromptContract):

    READS = ["query", "response"]
    WRITES = ["fact_candidates"]

    SYSTEM = """
    从对话中提取长期事实或用户偏好。
    每条必须包含:
    - key
    - type (preference / fact)
    - content
    - score (0-1)

    返回 JSON:
    {
      "fact_candidates": [
        {"key": "...", "type": "...", "content": "...", "score": 0.9}
      ]
    }
    """

    def build_user_prompt(self, state):
        return f"""
用户问题:
{state.get("query","")}

助手回答:
{state.get("response","")}
"""
