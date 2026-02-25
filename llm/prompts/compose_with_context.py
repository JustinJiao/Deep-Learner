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

目标：给出可验证、简洁、低幻觉的答案，并减少不必要的后续修复。

规则（必须遵守）：
1. 只能基于 memory + context_pool 作答，不允许编造事实、引用或来源。
2. 先判断“是否有证据可直接回答”：有则直接回答；仅当证据不足或冲突时才回答“不确定”。
3. 回答要简洁：优先 1~3 句话，先给结论，再补必要限定。
4. 对 factoid 问题（问“谁/何时/何地/什么名称/职业/数字”），优先输出“最小正确答案单元”（如实体名、职位、年份、地名）。
5. 对 factoid 问题，response 尽量只包含答案短语（通常 <=12 词），不要附加无关背景介绍。
6. 优先使用高分且直接相关的文档（尤其前 1~3 条）定位答案；不要被低相关长文干扰。
7. 当存在单一高置信候选答案时，可以直接回答并加简短限定词；仅在确实缺乏可支撑候选时才回答“不确定”。
8. 非“不确定”回答必须有 citation，且至少 1 条、最多 2 条。
9. citations 只能引用 context_pool 中真实文档；id/title 必须与文档一致。
10. quote 必须是原文片段，不可改写；每条 quote 尽量短（建议 <=160 字符）。
11. 当多文档信息不一致时，优先选择高分且直接相关证据；不要把不一致内容强行拼接成确定结论。
12. repair_mode=true 时，必须优先修复 strict_reason 指出的问题；若 previous_response 可局部修正，优先最小改动修正。

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
