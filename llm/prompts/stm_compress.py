# llm/prompts/stm_compress.py

from llm.prompts.base import PromptContract


class STMCompressPrompt(PromptContract):
    # 注意：这里读取的是我们在 stm_summary_node 里塞进去的 compress_chunk_text
    READS = ["short_term_memory", "compress_chunk_text"]
    WRITES = ["stm_compressed_text"]

    SYSTEM = """
你是 Deep-Learner 的短期记忆压缩器。
任务：把【待压缩轮】压缩成一段摘要，合并进【旧摘要】之后，输出新的摘要块。
要求：
- 必须保留可被问答检索的关键信息（如：用户偏好、地点、年龄、计划、结论、定义）。
- 不要编造，不要引入对话中没有的信息。
- Language requirement:
  - Output must be English-only.
  - Keep JSON keys unchanged; JSON string values must be in English.
- Project scope:
  - The conversation domain is limited to these filings:
    1) Amazon 10K 2024.pdf
    2) Alphabet 10K 2024.pdf
    3) MSFT 10-K.pdf
- 输出必须是 JSON，格式：
{"stm_compressed_text": "..."}
""".strip()

    def build_user_prompt(self, state):
        old = state.get("short_term_memory", "") or ""
        chunk = state.get("compress_chunk_text", "") or ""

        return f"""
【旧摘要】
{old}

【待压缩轮】
{chunk}

请输出新的摘要块（只输出 JSON）。
""".strip()
