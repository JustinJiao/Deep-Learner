import json 
import re 

def safe_parse_json (text :str ):
    """Super powerful JSON parser: handles line breaks, Markdown tags and unnecessary impurities"""
    if not text :
        raise ValueError ("Input text is empty")

        # 1. Try to remove Markdown code block tags
    cleaned =re .sub (r"```json\s*|\s*```","",text ).strip ()

    # 2. Try direct parsing (strict=False is key, allow literal newlines)
    try :
        return json .loads (cleaned ,strict =False )
    except json .JSONDecodeError :
    # 3. If that fails, try to regularly locate the outermost { and }
        match =re .search (r"(\{.*\})",cleaned ,re .DOTALL )
        if match :
            json_str =match .group (1 )
            try :
                return json .loads (json_str ,strict =False )
            except json .JSONDecodeError as e :
            # 4. If it still fails, it may be a quotation mark problem or structural damage.
                raise ValueError (f"JSON structure is corrupted and cannot be parsed: {e}")

    raise ValueError (f"Failed to extract a JSON object from text: {text[:100]}...")