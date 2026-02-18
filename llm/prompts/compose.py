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

    WRITES = ["response"]

    SYSTEM = """
    你是 Deep-Learner，一个严谨的技术导师。
    不允许编造事实。
    如果无法确定，请明确说明“不确定”。
    """

    HUMAN_TEMPLATE = """
    【历史摘要】
    {short_term_memory}

    【最近对话】
    {recent_messages}

    【长期记忆】
    {long_term_memory}

    【检索文档】
    {context_pool}

    【修复提示】
    {repair_hint}

    DirectPath: {is_direct_path}

    用户问题:
    {query}
    """

    def build_user_prompt(self, state):
        return self.HUMAN_TEMPLATE.format(
            short_term_memory=state.get("short_term_memory", ""),
            recent_messages=state.get("recent_messages", []),
            long_term_memory=state.get("long_term_memory", ""),
            context_pool=state.get("context_pool", []),
            repair_hint=state.get("repair_hint", ""),
            is_direct_path=state.get("is_direct_path", True),
            query=state.get("query", ""),
        )
