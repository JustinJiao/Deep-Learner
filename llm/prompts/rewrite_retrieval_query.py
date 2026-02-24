from llm.prompts.base import PromptContract


class RewriteRetrievalQueryPrompt(PromptContract):
    READS = ["query", "resolved_query", "short_term_memory", "recent_messages", "long_term_memory"]
    WRITES = ["retrieval_query"]

    SYSTEM = """
你是 Deep-Learner 的检索查询改写器。

任务：
1. 把用户问题改写为更容易命中文档检索的查询表达。
2. 可补充同义词、实体全称、关键限定词，但不得改变原意。
3. 优先保留用户的核心实体、时间、范围、否定条件。
4. 如果无法改得更好，retrieval_query 返回 resolved_query（或原 query）。

只输出 JSON：
{
  "retrieval_query": "..."
}
"""

    def build_user_prompt(self, state):
        return f"""
用户原始问题:
{state.get("query", "")}

指代消解后问题:
{state.get("resolved_query", "")}

短期记忆:
{state.get("short_term_memory", "")}

最近对话:
{state.get("recent_messages", [])}

长期记忆:
{state.get("long_term_memory", "")}
"""
