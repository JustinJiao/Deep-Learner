from utils.crypto_utils import generate_content_id

def split_into_semantic_chunks(structured_lines, chunk_size=800, overlap_size=150):
    """
    第二步：形态重塑
    功能：应用滑动窗口重叠和代码块保护协议
    """
    chunks = []
    buffer_lines = []
    curr_len = 0

    for item in structured_lines:
        buffer_lines.append(item['line'])
        curr_len += len(item['line'])

        # 触发切分条件：长度达标 且 不在代码块中
        if curr_len >= chunk_size and not item['is_code']:
            content = "\n".join(buffer_content := buffer_lines)
            chunks.append({
                "id": generate_content_id(content),
                "text": content,
                "metadata": {"h1": item['h1'], "h2": item['h2']}
            })
            
            # 实现重叠 (Overlap)：保留最后 N 行用于下一个块
            # 这里简单处理保留最后 3 行，可根据 overlap_size 动态计算
            buffer_lines = buffer_lines[-3:] if len(buffer_lines) > 3 else []
            curr_len = sum(len(l) for l in buffer_lines)

    # 兜底：处理剩余部分
    if buffer_lines:
        content = "\n".join(buffer_lines)
        chunks.append({
            "id": generate_content_id(content),
            "text": content,
            "metadata": {"h1": structured_lines[-1]['h1'], "h2": structured_lines[-1]['h2']}
        })
    return chunks