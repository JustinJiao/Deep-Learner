# llm/prompts/compose.py

from llm.prompts.base import PromptContract


class ComposePrompt(PromptContract):

    READS = [
        "query",
        "short_term_memory",
        "recent_messages",
        "is_direct_path",
        "rewritten_query",
        "context_pool",
        "long_term_memory",
        "repair_hint",
    ]

    WRITES = [
        "response",
        "citations",
    ]

    SYSTEM = """
你是 Deep-Learner，一个严谨的技术导师。

规则（必须遵守）：
1. 不允许编造事实。
2. 只能使用【长期记忆】或【检索文档】中的信息。
3. 如果无法确定，请明确说明“不确定”。
4. 如果使用了检索文档，必须在 citations 中列出来源。
5. citations 只能引用 context_pool 中提供的文档。
6. quote 必须是从 context_pool 中逐字复制的原文片段。
7. 不允许编造引用或改写引用。

你必须只输出 JSON。

JSON 格式：
{
  "response": "<最终回答文本>",
  "citations": [
    {
      "id": "<文档ID>",
      "title": "<文档标题>",
      "score": <相似度分数>,
      "quote": "<从原文中精确复制的一段文本>"
    }
  ]
}

如果没有使用任何检索文档：
"citations": []
"""

    HUMAN_TEMPLATE = """
【历史摘要】
{short_term_memory}

【最近对话】
{recent_messages}

【长期记忆】
{long_term_memory}

【检索文档列表】
{context_pool}

【修复提示】
{repair_hint}

DirectPath: {is_direct_path}

用户问题:
{query}
"""

    def build_user_prompt(self, state):

        docs = state.get("context_pool", [])

        formatted_docs = ""

        for i, d in enumerate(docs):
            formatted_docs += f"""
                文档 {i+1}
                ID: {d.get('id')}
                Title: {d.get('title')}
                Score: {d.get('score')}

                内容:
                {d.get('content')}

                --------------------
            """

        return self.HUMAN_TEMPLATE.format(
            short_term_memory=state.get("short_term_memory", ""),
            recent_messages=state.get("recent_messages", []),
            long_term_memory=state.get("long_term_memory", ""),
            context_pool=formatted_docs,
            repair_hint=state.get("repair_hint", ""),
            is_direct_path=state.get("is_direct_path", True),
            query=state.get("query", ""),
        )