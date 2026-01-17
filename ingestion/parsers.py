import os
from unstructured.partition.auto import partition
from unstructured.cleaners.core import clean, group_broken_paragraphs

class UniversalParser:
    def to_markdown(self, file_path):
        """
        策略：高精度解析 + 噪声清洗
        """
        # 执行布局分析，自动识别标题、文本和表格
        elements = partition(
            filename=file_path,
            strategy="hi_res",
            languages=["chi_sim", "eng"],
            infer_table_structure=True # 提取表格结构
        )
        
        md_lines = []
        for el in elements:
            # 执行归一化清洗
            text = clean(str(el), extra_whitespace=True, dashes=True)
            text = group_broken_paragraphs(text)
            
            if el.category == "Title":
                md_lines.append(f"## {text}")
            elif el.category == "Table":
                # 将表格存储为 HTML，保留结构化语义
                md_lines.append(el.metadata.text_as_html)
            else:
                md_lines.append(text)
                
        return "\n\n".join(md_lines)