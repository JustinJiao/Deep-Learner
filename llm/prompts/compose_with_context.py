from llm.prompts.base import PromptContract


class ComposeWithContextPrompt(PromptContract):
    READS = [
        "query",
        "short_term_memory",
        "recent_messages",
        "long_term_memory",
        "context_pool",
        "repair_mode",
        "strict_reason",
        "previous_response",
    ]
    WRITES = ["response", "citations"]

    SYSTEM = """
你是 Deep-Learner 的证据生成模块。

规则（必须遵守）：
1. 只能基于 memory + context_pool 作答。
2. 不允许编造事实与引用。
3. 如果 context_pool 中有可回答问题的证据，必须直接给出答案，不能输出“不确定”。
4. 只有在全部证据都无法支持答案或证据互相冲突时，才允许回答“不确定”。
5. 非“不确定”回答必须至少包含 1 条 citation。
6. citations 只能引用 context_pool 中文档。
7. quote 必须是原文摘录，不可改写。
8. repair_mode=true 时，优先修复 strict_reason 指出的错误。

必须只输出 JSON：
{
  "response": "...",
  "citations": [
    {
      "id": "...",
      "title": "...",
      "score": 0.0,
      "quote": "..."
    }
  ]
}
"""

    def build_user_prompt(self, state):
        docs = state.get("context_pool", [])
        formatted_docs = ""
        for i, d in enumerate(docs):
            formatted_docs += f"""
文档 {i + 1}
ID: {d.get('id')}
Title: {d.get('title')}
Score: {d.get('score')}
内容:
{d.get('content')}
--------------------
"""

        return f"""
repair_mode: {bool(state.get("repair_mode", False))}
strict_reason: {state.get("strict_reason", "")}
previous_response: {state.get("previous_response", "")}

用户问题:
{state.get("query", "")}

短期记忆:
{state.get("short_term_memory", "")}

最近对话:
{state.get("recent_messages", [])}

长期记忆:
{state.get("long_term_memory", "")}

检索文档:
{formatted_docs}
"""
