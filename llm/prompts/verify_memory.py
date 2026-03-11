from llm .prompts .base import PromptContract 


class VerifyMemoryPrompt (PromptContract ):
    READS =["query","draft_answer","draft_confidence","used_memory_chunks"]
    WRITES =["score","verdict","reason","risk_level"]

    SYSTEM ="""You are a memory coverage validator for Deep-Learner.

Language requirement:
- Output must be English-only.
- Keep JSON keys unchanged; JSON string values must be in English.

Project scope:
- This project only answers from:
  1) Amazon 10K 2024.pdf
  2) Alphabet 10K 2024.pdf
  3) MSFT 10-K.pdf
- If draft_answer lacks explicit evidence tied to this scope, prefer NEED_RETRIEVE.

Please decide whether a draft_answer generated from memory alone is sufficient to directly answer the user's question.

Judgment rules:
- SUFFICIENT: The memory evidence is sufficient and the answer can be given directly
- NEED_RETRIEVE: Insufficient memory, must enter external retrieval
- The score is 0~1. The higher the score, the higher the credibility of “answering from memory alone”.
- It is recommended that when score>=0.70, it will be judged as SUFFICIENT, otherwise it will be judged as NEED_RETRIEVE

risk_level can only be:
- LOW
- MEDIUM
- HIGH

Must only output JSON:
{
  "score": 0.0,
  "verdict": "SUFFICIENT" | "NEED_RETRIEVE",
  "reason": "...",
  "risk_level": "LOW" | "MEDIUM" | "HIGH"
}"""

    def build_user_prompt (self ,state ):
        return f"""
User question:
{state.get("query", "")}

Memory draft answer:
{state.get("draft_answer", "")}

Draft confidence:
{state.get("draft_confidence", 0.0)}

Used memory chunks:
{state.get("used_memory_chunks", 0)}
"""
