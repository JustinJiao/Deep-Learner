from llm.prompts.base import PromptContract


class ResolveQueryReferencePrompt(PromptContract):
    READS = ["query", "short_term_memory", "recent_messages", "long_term_memory"]
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
3. 可参考 long_term_memory 中的“项目背景事实”做轻量标准化，不改变原意：
   - 例如“most recent/latest fiscal year”可标准化为显式年份（若背景事实已给出）。
   - 例如“these companies/all three companies”可补全为明确公司名（若背景事实已给出）。
4. 不要引入背景事实之外的新实体/新数字；不得改变用户问题结论方向。
5. 如果 query 本身已明确，resolved_query 必须与原 query 等价。

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

长期记忆:
{state.get("long_term_memory", "")}
"""
