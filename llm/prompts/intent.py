# llm/prompts/intent.py

from llm.prompts.base import PromptContract


class IntentPrompt(PromptContract):

    READS = ["query", "short_term_memory", "recent_messages"]
    WRITES = ["intent"]

    SYSTEM = """
    判断用户问题属于哪种类型：
    - chat（闲聊/简单回答）
    - research（需要检索或推理）

    Language requirement:
    - Output must be English-only.
    - Keep JSON keys unchanged; JSON string values must be in English.

    Project scope:
    - This assistant is configured for these filings only:
      1) Amazon 10K 2024.pdf
      2) Alphabet 10K 2024.pdf
      3) MSFT 10-K.pdf
    - Finance/risk/business questions about these companies should be classified as research.

    必须返回 JSON 格式：
    {
        "intent": {
            "type": "chat" 或 "research"
        }
    }
    """

    def build_user_prompt(self, state):
        return f"""
历史摘要:
{state.get("short_term_memory","")}

最近对话:
{state.get("recent_messages",[])}

用户问题:
{state.get("query","")}
"""
