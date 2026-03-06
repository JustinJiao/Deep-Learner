import hashlib
import os
import re
from config.settings import AppConfig


class SemanticChunker:
    HTML_TABLE_MARKERS = ("<table", "</table>", "<tr")
    MD_TABLE_SEPARATOR_RE = re.compile(
        r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$"
    )

    def __init__(self, chunk_size=None, overlap_lines=None):
        self.chunk_size = chunk_size if chunk_size is not None else AppConfig.CHUNK_SIZE
        self.overlap_lines = (
            overlap_lines if overlap_lines is not None else AppConfig.CHUNK_OVERLAP
        )

    def generate_id(self, content):
        return hashlib.md5(content.encode()).hexdigest()

    def _is_html_table_line(self, line: str) -> bool:
        lowered = line.lower()
        return any(marker in lowered for marker in self.HTML_TABLE_MARKERS)

    def _is_markdown_table_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if "|" not in stripped:
            return False
        if self.MD_TABLE_SEPARATOR_RE.match(stripped):
            return True
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            return True
        return False

    def _table_mode(self, line: str) -> str:
        if self._is_html_table_line(line):
            return "html"
        if self._is_markdown_table_line(line):
            return "markdown"
        return ""

    def split_with_overlap(self, text, source_path):
        lines = text.split("\n")
        chunks = []
        buffer_lines = []
        curr_len, is_code = 0, False
        h1 = os.path.basename(source_path)
        h2 = "Intro"
        table_lines = []
        in_table = False
        table_mode = ""
        table_index = 0

        def flush_normal(with_overlap: bool) -> None:
            nonlocal buffer_lines, curr_len
            if not buffer_lines:
                return

            content = "\n".join(buffer_lines).strip("\n")
            if content.strip():
                chunks.append(
                    self._build_chunk(
                        content,
                        h1,
                        h2,
                        source_path,
                        is_table=False,
                    )
                )

            if with_overlap and self.overlap_lines > 0:
                if len(buffer_lines) > self.overlap_lines:
                    buffer_lines = buffer_lines[-self.overlap_lines :]
                else:
                    buffer_lines = []
                curr_len = sum(len(line) for line in buffer_lines)
            else:
                buffer_lines = []
                curr_len = 0

        def flush_table() -> None:
            nonlocal table_lines, table_index, table_mode
            if not table_lines:
                return

            content = "\n".join(table_lines).strip("\n")
            if content.strip():
                table_index += 1
                chunks.append(
                    self._build_chunk(
                        content,
                        h1,
                        h2,
                        source_path,
                        is_table=True,
                        table_index=table_index,
                    )
                )
            table_lines = []
            table_mode = ""

        for line in lines:
            if line.strip().startswith("```"):
                is_code = not is_code
            if line.startswith("## "):
                h2 = line.replace("## ", "", 1).strip()

            line_mode = self._table_mode(line) if not is_code else ""

            if in_table:
                if table_mode == "html":
                    table_lines.append(line)
                    if "</table>" in line.lower():
                        flush_table()
                        in_table = False
                    continue
                if table_mode == "markdown":
                    if self._is_markdown_table_line(line):
                        table_lines.append(line)
                        continue
                    flush_table()
                    in_table = False

            if line_mode:
                flush_normal(with_overlap=False)
                in_table = True
                table_mode = line_mode
                table_lines = [line]
                if table_mode == "html" and "</table>" in line.lower():
                    flush_table()
                    in_table = False
                continue

            buffer_lines.append(line)
            curr_len += len(line)

            if curr_len >= self.chunk_size and not is_code:
                flush_normal(with_overlap=True)

        if in_table:
            flush_table()
        if buffer_lines:
            flush_normal(with_overlap=False)
        return chunks

    def _build_chunk(
        self,
        content,
        h1,
        h2,
        source,
        is_table=False,
        table_index=None,
    ):
        metadata = {
            "h1": h1,
            "h2": h2,
            "source": source,
            "is_table": bool(is_table),
        }
        if is_table and table_index is not None:
            metadata["table_index"] = int(table_index)

        return {
            "doc_id": self.generate_id(content),
            "content": content,
            "metadata": metadata,
        }
