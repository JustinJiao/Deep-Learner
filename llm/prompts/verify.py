# llm/prompts/verify.py

from llm .prompts .base import PromptContract 


class VerifyPrompt (PromptContract ):

    READS =["response","context_pool"]
    WRITES =["score","is_hallucination","critique","error_type","next_step"]

    SYSTEM ="""Check if the answer is based on the retrieved document.
    Language requirement:
    - Output must be English-only.
    - Keep JSON keys unchanged; JSON string values must be in English.

    Project scope:
    - Validate only against these filings:
      1) Amazon 10K 2024.pdf
      2) Alphabet 10K 2024.pdf
      3) MSFT 10-K.pdf
    - Claims outside these filings should be treated as unsupported.

    If there is a problem, please classify the error type:
    - generation_error
    -retrieval_insufficient
    -query_misaligned

    And give the next step:
    - compose
    -retrieve
    -query_rewrite

    Output rules (must be followed):
    - The score is 0~1. The higher the score, the more reliable the answer is and the more it can be supported by evidence.
    - is_hallucination should be false when score >= 0.68; otherwise it should be true.
    - When is_hallucination = false, error_type must be an empty string and next_step must be an empty string.
    - When is_hallucination = true, error_type and next_step must be filled in with the above enumeration values.

    Return JSON:
    {
      "score": 0.0,
      "is_hallucination": true/false,
      "critique": "...",
      "error_type": "...",
      "next_step": "..."
    }"""

    def build_user_prompt (self ,state ):
        return f"""
Answer:
{state.get("response","")}

Retrieved documents:
{state.get("context_pool",[])}
"""
