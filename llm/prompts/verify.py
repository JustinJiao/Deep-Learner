# llm/prompts/verify.py

from llm.prompts.base import PromptContract


class VerifyPrompt(PromptContract):

    READS = ["response", "context_pool"]
    WRITES = ["is_hallucination", "critique", "error_type", "next_step"]

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

    返回 JSON:
    {
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
