from llm.prompts.base import PromptContract


class ComposeMemoryDraftPrompt(PromptContract):
    READS = ["query", "short_term_memory", "recent_messages", "long_term_memory"]
    WRITES = ["draft_answer", "confidence", "used_memory_chunks"]

    SYSTEM = """
你是 Deep-Learner 的记忆草答模块。

任务：
1. 只能基于短期记忆（STM）和长期记忆（LTM）草拟回答。
2. 如果记忆不足，明确表达不确定。
3. 估计本次草答的可信度 confidence（0~1）。
4. used_memory_chunks 表示本次草答实际引用到的记忆片段数量（整数）。

必须只输出 JSON，格式如下：
{
  "draft_answer": "...",
  "confidence": 0.0,
  "used_memory_chunks": 0
}
"""

    def build_user_prompt(self, state):
        return f"""
用户问题:
{state.get("query", "")}

短期记忆:
{state.get("short_term_memory", "")}

最近对话:
{state.get("recent_messages", [])}

长期记忆:
{state.get("long_term_memory", "")}
"""
