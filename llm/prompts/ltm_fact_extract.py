# llm/prompts/ltm_fact_extract.py

from llm .prompts .base import PromptContract 


class LTMFactExtractPrompt (PromptContract ):

    READS =["query","response"]
    WRITES =["fact_candidates"]

    SYSTEM ="""Extract long-term facts or user preferences from conversations.
    Each entry must contain:
    - key
    - type (preference / fact)
    - content
    - score (0-1)

    Language requirement:
    - Output must be English-only.
    - Keep JSON keys unchanged; JSON string values must be in English.

    Project scope:
    - Extract only facts/preferences relevant to QA over:
      1) Amazon 10K 2024.pdf
      2) Alphabet 10K 2024.pdf
      3) MSFT 10-K.pdf

    Return JSON:
    {
      "fact_candidates": [
        {"key": "...", "type": "...", "content": "...", "score": 0.9}
      ]
    }"""

    def build_user_prompt (self ,state ):
        return f"""
User question:
{state.get("query","")}

Assistant answer:
{state.get("response","")}
"""
