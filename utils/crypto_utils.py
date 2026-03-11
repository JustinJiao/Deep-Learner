import hashlib 
def generate_content_id (content :str )->str :
    """Solve ID alignment problem: generate unique MD5 based on content"""
    return hashlib .md5 (content .encode ('utf-8')).hexdigest ()