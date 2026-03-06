from __future__ import annotations

from pathlib import Path

from unstructured.cleaners.core import clean, group_broken_paragraphs
from unstructured.partition.auto import partition

from config.settings import AppConfig

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None

class UniversalParser:
    @staticmethod
    def _normalize_cell(value: object) -> str:
        text = "" if value is None else str(value)
        text = text.replace("\n", " ").strip()
        # 避免破坏 markdown 表格结构
        return text.replace("|", "\\|")

    def _table_to_markdown(self, raw_table: list[list[object]]) -> str:
        if not raw_table:
            return ""

        rows: list[list[str]] = []
        max_cols = 0
        for row in raw_table:
            normalized = [self._normalize_cell(x) for x in (row or [])]
            if any(cell for cell in normalized):
                rows.append(normalized)
                max_cols = max(max_cols, len(normalized))

        if not rows or max_cols == 0:
            return ""

        def pad(r: list[str]) -> list[str]:
            if len(r) < max_cols:
                return r + [""] * (max_cols - len(r))
            return r

        rows = [pad(r) for r in rows]
        header = rows[0]
        if not any(cell.strip() for cell in header):
            header = [f"col_{i + 1}" for i in range(max_cols)]
            body = rows
        else:
            body = rows[1:]

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * max_cols) + " |",
        ]
        for r in body:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)

    def _extract_pdf_tables_markdown(self, file_path: str) -> list[str]:
        if pdfplumber is None:
            return []

        max_pages = max(0, int(AppConfig.PDF_PDFPLUMBER_MAX_PAGES))
        table_settings = {
            "vertical_strategy": AppConfig.PDFPLUMBER_VERTICAL_STRATEGY,
            "horizontal_strategy": AppConfig.PDFPLUMBER_HORIZONTAL_STRATEGY,
        }
        results: list[str] = []
        with pdfplumber.open(file_path) as pdf:
            pages = pdf.pages if max_pages == 0 else pdf.pages[:max_pages]
            for page_idx, page in enumerate(pages, 1):
                tables = page.extract_tables(table_settings=table_settings) or []
                for table_idx, table in enumerate(tables, 1):
                    table_md = self._table_to_markdown(table)
                    if not table_md:
                        continue
                    results.append(f"## Table P{page_idx}-{table_idx}\n{table_md}")
        return results

    def _text_to_markdown(self, file_path: str) -> str:
        # fast 文本解析（策略可由环境变量覆盖）
        elements = partition(
            filename=file_path,
            strategy=AppConfig.UNSTRUCTURED_STRATEGY,
            languages=AppConfig.UNSTRUCTURED_LANGUAGES,
            infer_table_structure=AppConfig.UNSTRUCTURED_INFER_TABLE_STRUCTURE,
        )

        md_lines = []
        for el in elements:
            text = clean(str(el), extra_whitespace=True, dashes=True)
            text = group_broken_paragraphs(text)

            if el.category == "Title":
                md_lines.append(f"## {text}")
            elif el.category == "Table":
                # 兼容非 PDF 或 fallback 场景
                md_lines.append(text)
            else:
                md_lines.append(text)
        return "\n\n".join(md_lines)

    def to_markdown(self, file_path):
        text_md = self._text_to_markdown(file_path)
        is_pdf = Path(file_path).suffix.lower() == ".pdf"

        if not is_pdf or not AppConfig.PDF_EXTRACT_TABLES_WITH_PDFPLUMBER:
            return text_md

        table_blocks = self._extract_pdf_tables_markdown(file_path)
        if not table_blocks:
            return text_md

        return (
            f"{text_md}\n\n"
            "## Extracted Tables (pdfplumber)\n\n"
            + "\n\n".join(table_blocks)
        )
