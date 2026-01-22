import hashlib
import os
from config.settings import AppConfig

class SemanticChunker:
    def __init__(self, chunk_size=None, overlap_lines=None):
        self.chunk_size = chunk_size or AppConfig.CHUNK_SIZE
        self.overlap_lines = overlap_lines or AppConfig.CHUNK_OVERLAP

    def generate_id(self, content):
        return hashlib.md5(content.encode()).hexdigest()

    def split_with_overlap(self, text, source_path):
        lines = text.split("\n")
        chunks, buffer_lines = [], []
        curr_len, is_code = 0, False
        h1 = os.path.basename(source_path)
        h2 = "Intro"

        for line in lines:
            if line.strip().startswith("```"): is_code = not is_code
            if line.startswith("## "): h2 = line.replace("## ", "").strip()

            buffer_lines.append(line)
            curr_len += len(line)

            if curr_len >= self.chunk_size and not is_code:
                content = "\n".join(buffer_lines)
                chunks.append(self._build_chunk(content, h1, h2, source_path))
                buffer_lines = buffer_lines[-self.overlap_lines:] if len(buffer_lines) > self.overlap_lines else []
                curr_len = sum(len(l) for l in buffer_lines)

        if buffer_lines:
            chunks.append(self._build_chunk("\n".join(buffer_lines), h1, h2, source_path))
        return chunks

    def _build_chunk(self, content, h1, h2, source):
        # 🌟 关键修改：将离散字段打包进 metadata 字典
        return {
            "doc_id": self.generate_id(content),
            "content": content,
            "metadata": {
                "h1": h1,
                "h2": h2,
                "source": source
            }
        }