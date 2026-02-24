# llm/prompts/verify.py

from llm.prompts.base import PromptContract


class VerifyPrompt(PromptContract):

    READS = ["response", "context_pool"]
    WRITES = ["score", "is_hallucination", "critique", "error_type", "next_step"]

    SYSTEM = """
    检查回答是否基于检索文档。
    如果有问题，请分类错误类型：
    - generation_error
    - retrieval_insufficient
    - query_misaligned

    并给出下一步：
    - compose
    - retrieve
    - query_rewrite

    输出规则（必须遵守）：
    - score 为 0~1，分数越高表示回答越可靠、越可被证据支持。
    - 当 score >= 0.68 时，is_hallucination 应为 false；否则应为 true。
    - 当 is_hallucination = false 时，error_type 必须是空字符串，next_step 必须是空字符串。
    - 当 is_hallucination = true 时，error_type 和 next_step 必须填写为上面的枚举值。

    返回 JSON:
    {
      "score": 0.0,
      "is_hallucination": true/false,
      "critique": "...",
      "error_type": "...",
      "next_step": "..."
    }
    """

    def build_user_prompt(self, state):
        return f"""
回答:
{state.get("response","")}

检索文档:
{state.get("context_pool",[])}
"""
