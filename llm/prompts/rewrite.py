# llm/prompts/rewrite.py

from llm.prompts.base import PromptContract


class RewritePrompt(PromptContract):

    READS = ["query", "short_term_memory", "recent_messages", "long_term_memory"]
    WRITES = ["rewritten_query"]

    SYSTEM = """
    将用户问题改写为更适合检索的查询。
    不要改变原意。
    Language requirement:
    - Output must be English-only.
    - Keep JSON keys unchanged; JSON string values must be in English.
    Project scope:
    - Keep rewrite within this closed corpus:
      1) Amazon 10K 2024.pdf
      2) Alphabet 10K 2024.pdf
      3) MSFT 10-K.pdf
    返回 JSON: {"rewritten_query": "..."}
    """

    def build_user_prompt(self, state):
        return f"""
用户问题:
{state.get("query","")}

长期记忆:
{state.get("long_term_memory","")}
"""
