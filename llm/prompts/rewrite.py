# llm/prompts/rewrite.py

from llm .prompts .base import PromptContract 


class RewritePrompt (PromptContract ):

    READS =["query","short_term_memory","recent_messages","long_term_memory"]
    WRITES =["rewritten_query"]

    SYSTEM ="""Reword user questions into queries more suitable for retrieval.
    Don't change your original intention.
    Language requirement:
    - Output must be English-only.
    - Keep JSON keys unchanged; JSON string values must be in English.
    Project scope:
    - Keep rewrite within this closed corpus:
      1) Amazon 10K 2024.pdf
      2) Alphabet 10K 2024.pdf
      3) MSFT 10-K.pdf
    Return JSON: {"rewritten_query": "..."}"""

    def build_user_prompt (self ,state ):
        return f"""
User question:
{state.get("query","")}

Long-term memory:
{state.get("long_term_memory","")}
"""
