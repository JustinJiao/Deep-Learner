from llm.prompts.base import PromptContract


class ResolveQueryReferencePrompt(PromptContract):
    READS = ["query", "short_term_memory", "recent_messages"]
    WRITES = ["resolved_query"]

    SYSTEM = """
你是 Deep-Learner 的指代消解器。

Language requirement:
- Output must be English-only.
- Keep JSON keys unchanged; JSON string values must be in English.

Project scope:
- This project only serves QA for these filings:
  1) Amazon 10K 2024.pdf
  2) Alphabet 10K 2024.pdf
  3) MSFT 10-K.pdf
- Resolve pronouns/references within this scope only.

任务：
1. 识别用户 query 中的指代词（如“它/这个/上一个问题/那件事”）。
2. 结合 recent_messages 与 short_term_memory，把 query 改写为语义明确的完整问题。
3. 只能做“指代补全”，不能扩展检索关键词，不能改变原意。
4. 如果 query 本身已明确，resolved_query 必须与原 query 等价。

只输出 JSON：
{
  "resolved_query": "..."
}
"""

    def build_user_prompt(self, state):
        return f"""
用户当前问题:
{state.get("query", "")}

短期记忆:
{state.get("short_term_memory", "")}

最近对话:
{state.get("recent_messages", [])}
"""
