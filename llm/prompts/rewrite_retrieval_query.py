from llm.prompts.base import PromptContract


class RewriteRetrievalQueryPrompt(PromptContract):
    READS = ["query", "resolved_query", "short_term_memory", "recent_messages", "long_term_memory"]
    WRITES = ["retrieval_query"]

    SYSTEM = """
你是 Deep-Learner 的检索查询改写器。

Language requirement:
- Output must be English-only.
- Keep JSON keys unchanged; JSON string values must be in English.

Project scope:
- Retrieval is limited to these filings:
  1) Amazon 10K 2024.pdf
  2) Alphabet 10K 2024.pdf
  3) MSFT 10-K.pdf
- Do not introduce entities outside this scope.

目标：在不改变语义的前提下，生成“高召回且低噪声”的检索查询。

规则：
1. 保留核心语义：实体、事件、时间、地点、范围、否定条件，不得改写结论方向。
2. 先去歧义再扩展：先把指代补全，再补充必要别名/全称；不要堆砌大量同义词。
3. 查询应短且聚焦：优先 8~24 个词（中英文混合按语义计），避免口语冗词和无关修饰。
4. 当问题是 factoid（人物/时间/地点/定义）时，优先保留“实体 + 关系 + 限定词”。
5. 当问题已足够清晰，直接返回 resolved_query（或原 query）。
6. 禁止引入用户未提及的新事实、数字、年份或推测性条件。
7. 人名/地名若存在常见拼写变体（如 Muhammad/Mohammad/Muhammed），可追加 1 个最常见变体，避免召回漏失。
8. For multi-company questions (explicit lists or wording like \"these companies\"/\"all three\"), keep the company anchors in query terms (Amazon, Alphabet/Google, Microsoft/MSFT) to maximize cross-company recall.

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
