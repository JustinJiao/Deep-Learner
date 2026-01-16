import hashlib
def generate_content_id(content: str) -> str:
    """解决 ID 对齐问题：基于内容生成唯一 MD5"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()