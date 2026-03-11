# llm/prompts/intent.py

from llm .prompts .base import PromptContract 


class IntentPrompt (PromptContract ):

    READS =["query","short_term_memory","recent_messages"]
    WRITES =["intent"]

    SYSTEM ="""Determine what type of user problem it is:
    - chat (small talk/simple answer)
    - research (requires retrieval or reasoning)

    Language requirement:
    - Output must be English-only.
    - Keep JSON keys unchanged; JSON string values must be in English.

    Project scope:
    - This assistant is configured for these filings only:
      1) Amazon 10K 2024.pdf
      2) Alphabet 10K 2024.pdf
      3) MSFT 10-K.pdf
    - Finance/risk/business questions about these companies should be classified as research.

    Must return JSON format:
    {
        "intent": {
            "type": "chat" or "research"
        }
    }"""

    def build_user_prompt (self ,state ):
        return f"""
Historical summary:
{state.get("short_term_memory","")}

Recent messages:
{state.get("recent_messages",[])}

User question:
{state.get("query","")}
"""
