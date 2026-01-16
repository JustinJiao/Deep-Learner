def parse_markdown_structure(content: str):
    """
    第一步：清洗与结构识别
    功能：提取 H1, H2 并识别代码块边界
    """
    lines = content.split('\n')
    structured_lines = []
    curr_h1, curr_h2 = "Unknown", "General"
    in_code_block = False

    for line in lines:
        if line.startswith('# '):
            curr_h1 = line.replace('# ', '').strip()
            continue
        elif line.startswith('## '):
            curr_h2 = line.replace('## ', '').strip()
            continue
        
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
        
        # 将每一行与其所属的元数据绑定
        structured_lines.append({
            "line": line,
            "h1": curr_h1,
            "h2": curr_h2,
            "is_code": in_code_block
        })
    return structured_lines