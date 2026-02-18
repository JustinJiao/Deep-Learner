# llm/prompts/rewrite.py

from llm.prompts.base import PromptContract


class RewritePrompt(PromptContract):

    READS = ["query", "short_term_memory", "recent_messages", "long_term_memory"]
    WRITES = ["rewritten_query"]

    SYSTEM = """
    将用户问题改写为更适合检索的查询。
    不要改变原意。
    返回 JSON: {"rewritten_query": "..."}
    """

    def build_user_prompt(self, state):
        return f"""
用户问题:
{state.get("query","")}

长期记忆:
{state.get("long_term_memory","")}
"""
