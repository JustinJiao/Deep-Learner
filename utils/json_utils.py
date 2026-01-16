import json
import re

def safe_parse_json(text: str):
    """
    超强力 JSON 解析器：处理换行、Markdown 标签及多余杂质
    """
    if not text:
        raise ValueError("输入文本为空")

    # 1. 尝试移除 Markdown 代码块标记
    cleaned = re.sub(r"```json\s*|\s*```", "", text).strip()
    
    # 2. 尝试直接解析 (strict=False 是关键，允许 literal newlines)
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        # 3. 如果失败，尝试正则定位最外层的 { 和 }
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str, strict=False)
            except json.JSONDecodeError as e:
                # 4. 如果还是失败，可能是引号问题或结构损毁
                raise ValueError(f"JSON 结构损毁，无法解析: {e}")
        
    raise ValueError(f"未能从文本中提取到 JSON 对象: {text[:100]}...")