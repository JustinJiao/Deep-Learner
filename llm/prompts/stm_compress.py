# llm/prompts/stm_compress.py

from llm .prompts .base import PromptContract 


class STMCompressPrompt (PromptContract ):
# Note: What is read here is the compress_chunk_text we put in stm_summary_node
    READS =["short_term_memory","compress_chunk_text"]
    WRITES =["stm_compressed_text"]

    SYSTEM ="""You are Deep-Learner’s short-term memory compressor.
Task: Compress the [round to be compressed] into a summary, merge it into the [old summary], and output a new summary block.
Requirements:
- Key information that can be retrieved by Q&A must be retained (e.g. user preferences, location, age, plans, conclusions, definitions).
- Don’t make it up, don’t introduce information that is not in the conversation.
- Language requirement:
  - Output must be English-only.
  - Keep JSON keys unchanged; JSON string values must be in English.
- Project scope:
  - The conversation domain is limited to these filings:
    1) Amazon 10K 2024.pdf
    2) Alphabet 10K 2024.pdf
    3) MSFT 10-K.pdf
- Output must be JSON, format:
{"stm_compressed_text": "..."}""".strip ()

    def build_user_prompt (self ,state ):
        old =state .get ("short_term_memory","")or ""
        chunk =state .get ("compress_chunk_text","")or ""

        return f"""
[Old summary]
{old}

[Rounds to compress]
{chunk}

Please output the new summary block (JSON only).
""".strip ()
