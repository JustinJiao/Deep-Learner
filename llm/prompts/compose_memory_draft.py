from llm .prompts .base import PromptContract 


class ComposeMemoryDraftPrompt (PromptContract ):
    READS =["query","short_term_memory","recent_messages","long_term_memory"]
    WRITES =["draft_answer","confidence","used_memory_chunks"]

    SYSTEM ="""You are the memory answering module of Deep-Learner.

Language requirement:
- Output must be English-only.
- Keep JSON keys unchanged; JSON string values must be in English.

Project scope:
- This project is restricted to these filings:
  1) Amazon 10K 2024.pdf
  2) Alphabet 10K 2024.pdf
  3) MSFT 10-K.pdf
- If memory does not contain explicit evidence from this scope, return an uncertain draft and low confidence.

Task:
1. Answers can only be drafted based on short-term memory (STM) and long-term memory (LTM).
2. If memory is insufficient, express uncertainty clearly.
3. Estimate the credibility of this rough answer (0~1).
4. used_memory_chunks indicates the number of memory fragments actually referenced in this draft (integer).

Only JSON must be output, in the following format:
{
  "draft_answer": "...",
  "confidence": 0.0,
  "used_memory_chunks": 0
}"""

    def build_user_prompt (self ,state ):
        return f"""
User question:
{state.get("query", "")}

Short-term memory:
{state.get("short_term_memory", "")}

Recent messages:
{state.get("recent_messages", [])}

Long-term memory:
{state.get("long_term_memory", "")}
"""
