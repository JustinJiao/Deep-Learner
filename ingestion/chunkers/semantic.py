import hashlib 
import os 
import re 
from config .settings import AppConfig 


class SemanticChunker :
    HTML_TABLE_MARKERS =("<table","</table>","<tr")
    MD_TABLE_SEPARATOR_RE =re .compile (
    r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$"
    )

    def __init__ (self ,chunk_size =None ,overlap_lines =None ):
        self .chunk_size =chunk_size if chunk_size is not None else AppConfig .CHUNK_SIZE 
        self .overlap_lines =(
        overlap_lines if overlap_lines is not None else AppConfig .CHUNK_OVERLAP 
        )
        self .table_row_fact_only =bool (AppConfig .TABLE_CHUNK_ROW_FACT_ONLY )
        self .table_window_rows =max (1 ,int (AppConfig .TABLE_CHUNK_WINDOW_ROWS ))
        self .emit_table_window_chunk =bool (AppConfig .TABLE_CHUNK_EMIT_WINDOW )and (not self .table_row_fact_only )
        self .emit_table_row_facts =bool (AppConfig .TABLE_CHUNK_EMIT_ROW_FACTS )or self .table_row_fact_only 
        self .emit_table_raw_chunk =bool (AppConfig .TABLE_CHUNK_EMIT_RAW )and (not self .table_row_fact_only )
        self .table_max_row_facts =max (1 ,int (AppConfig .TABLE_CHUNK_MAX_ROW_FACTS ))

    def generate_id (self ,content ):
        return hashlib .md5 (content .encode ()).hexdigest ()

    def _is_html_table_line (self ,line :str )->bool :
        lowered =line .lower ()
        return any (marker in lowered for marker in self .HTML_TABLE_MARKERS )

    def _is_markdown_table_line (self ,line :str )->bool :
        stripped =line .strip ()
        if not stripped :
            return False 
        if "|"not in stripped :
            return False 
        if self .MD_TABLE_SEPARATOR_RE .match (stripped ):
            return True 
        if stripped .startswith ("|")and stripped .endswith ("|")and stripped .count ("|")>=2 :
            return True 
        return False 

    def _table_mode (self ,line :str )->str :
        if self ._is_html_table_line (line ):
            return "html"
        if self ._is_markdown_table_line (line ):
            return "markdown"
        return ""

    @staticmethod 
    def _normalize_cell (text :str )->str :
        cell =str (text or "").replace ("\\|","|").strip ()
        return re .sub (r"\s+"," ",cell )

    def _split_markdown_row (self ,line :str )->list [str ]:
        stripped =str (line or "").strip ()
        if stripped .startswith ("|"):
            stripped =stripped [1 :]
        if stripped .endswith ("|"):
            stripped =stripped [:-1 ]
        return [self ._normalize_cell (c )for c in stripped .split ("|")]

    def _is_markdown_separator_row (self ,line :str )->bool :
        return bool (self .MD_TABLE_SEPARATOR_RE .match (str (line or "").strip ()))

    def _format_markdown_row (self ,cells :list [str ],col_count :int )->str :
        padded =list (cells [:col_count ])+[""]*max (0 ,col_count -len (cells ))
        return "| "+" | ".join (padded )+" |"

    @staticmethod 
    def _year_token_count (values :list [str ])->int :
        count =0 
        for cell in values :
            if re .search (r"\b20\d{2}\b",str (cell or "")):
                count +=1 
        return count 

    @staticmethod 
    def _header_quality (values :list [str ])->bool :
        nonempty =[str (v or "").strip ()for v in values if str (v or "").strip ()]
        if len (nonempty )<2 :
            return False 
        alpha_cells =0 
        numeric_like =0 
        for cell in nonempty :
            if re .search (r"[A-Za-z]{2,}",cell ):
                alpha_cells +=1 
            if re .fullmatch (r"[\d,.\-()%$]+",cell ):
                numeric_like +=1 
        if alpha_cells <2 :
            return False 
        if numeric_like >=alpha_cells :
            return False 
        return True 

    def _parse_markdown_table (self ,table_lines :list [str ])->tuple [list [str ],list [list [str ]]]:
        parsed_rows :list [list [str ]]=[]
        for line in table_lines :
            if not self ._is_markdown_table_line (line ):
                continue 
            if self ._is_markdown_separator_row (line ):
                continue 
            row =self ._split_markdown_row (line )
            if any (cell .strip ()for cell in row ):
                parsed_rows .append (row )

        if not parsed_rows :
            return [],[]

        max_cols =max (len (r )for r in parsed_rows )
        normalized_rows :list [list [str ]]=[]
        for row in parsed_rows :
            normalized_rows .append (row +[""]*(max_cols -len (row )))

        header =normalized_rows [0 ]
        body =normalized_rows [1 :]if len (normalized_rows )>1 else []
        if not self ._header_quality (header ):
            year_header_idx =-1 
            for idx ,row in enumerate (normalized_rows [:min (6 ,len (normalized_rows ))]):
                if self ._year_token_count (row )>=2 :
                    year_header_idx =idx 
                    break 

            if year_header_idx >=0 :
                header =normalized_rows [year_header_idx ]
                body =[
                row 
                for idx ,row in enumerate (normalized_rows )
                if idx !=year_header_idx 
                ]
            else :
                header =[f"col_{idx + 1}"for idx in range (max_cols )]
                body =normalized_rows 
        return header ,body 

    def _build_row_fact (
    self ,
    h1 :str ,
    h2 :str ,
    table_index :int ,
    row_index :int ,
    header :list [str ],
    row :list [str ],
    )->str :
        def join_parts (parts :list [str ])->str :
            cleaned =[str (p ).strip ()for p in parts if str (p ).strip ()]
            if not cleaned :
                return ""
            if len (cleaned )==1 :
                return cleaned [0 ]
            if len (cleaned )==2 :
                return f"{cleaned[0]} and {cleaned[1]}"
            return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"

        def clean_label (label :str ,idx :int )->str :
            raw =self ._normalize_cell (label )
            if not raw :
                raw =f"column {idx + 1}"
            lowered =raw .lower ()
            if re .fullmatch (r"col_\d+",lowered ):
                return f"column {idx + 1}"
            raw =re .sub (r"[_:]+"," ",raw )
            raw =re .sub (r"\s+"," ",raw ).strip (" -;,.")
            metric_aliases =(
            "net sales",
            "revenue",
            "growth",
            "operating income",
            "income",
            "assets",
            )
            lowered =raw .lower ()
            for alias in metric_aliases :
                if alias in lowered :
                    return alias 
            return raw or f"column {idx + 1}"

        entries :list [dict ]=[]
        for idx ,value in enumerate (row ):
            value_text =self ._normalize_cell (value )
            if not value_text :
                continue 
            key =clean_label (header [idx ]if idx <len (header )else "",idx )
            year_match =re .search (r"\b(20\d{2})\b",key )
            year =int (year_match .group (1 ))if year_match else None 
            entries .append (
            {
            "idx":idx ,
            "key":key ,
            "value":value_text ,
            "year":year ,
            }
            )

        if not entries :
            return ""

        metric_hint =""
        table_context =f"{h1} {h2}".lower ()
        for alias in (
        "revenue",
        "net sales",
        "operating income",
        "income",
        "growth",
        "assets",
        ):
            if alias in table_context :
                metric_hint =alias 
                break 

        subject =""
        subject_idx =-1 
        for item in entries :
            value_text =str (item ["value"])
            if re .search (r"[A-Za-z]{2,}",value_text )and not re .fullmatch (r"[\d,.$()%\- ]+",value_text ):
                subject =value_text 
                subject_idx =int (item ["idx"])
                break 
        if not subject :
            subject ="This row"

        year_values :list [tuple [int ,str ]]=[]
        clauses :list [str ]=[]
        for item in entries :
            idx =int (item ["idx"])
            if idx ==subject_idx :
                continue 
            key =str (item ["key"])
            value_text =str (item ["value"])
            year =item ["year"]
            if year is not None :
                year_values .append ((int (year ),value_text ))
                continue 
            clean_key =re .sub (r"\s+"," ",key ).strip ().lower ()
            if not clean_key :
                clean_key ="value"
            if clean_key .startswith ("column "):
                clean_key ="value"
            clauses .append (f"{clean_key} is {value_text}")

        if year_values :
            ordered =sorted (year_values ,key =lambda x :x [0 ])
            if metric_hint :
                clauses .extend (
                [
                f"{metric_hint} in {year} is {value}"
                for year ,value in ordered 
                ]
                )
            else :
                clauses .extend ([f"value in {year} is {value}"for year ,value in ordered ])

        if not clauses :
            fallback_value =str (entries [0 ]["value"])
            clauses =[f"value is {fallback_value}"]

        limited_clauses =clauses [:8 ]
        detail =join_parts (limited_clauses )
        subject_prefix ="This row"if subject =="This row"else f"{subject} company"
        text =f"{subject_prefix} {detail}."
        text +=f" Source section is {h1} [{h2}], table {table_index}, row {row_index}."
        return text 

    @staticmethod 
    def _row_fact_row_is_valid (header :list [str ],row :list [str ])->bool :
        nonempty =[str (v or "").strip ()for v in row if str (v or "").strip ()]
        if len (nonempty )<2 :
            return False 
        if len (nonempty )>16 :
            return False 
        numeric_cells =0 
        alpha_cells =0 
        for value in nonempty :
            if re .search (r"\d",value ):
                numeric_cells +=1 
            if re .search (r"[A-Za-z]{2,}",value ):
                alpha_cells +=1 
        if numeric_cells <=0 :
            return False 
        if alpha_cells <=0 :
            return False 

        meaningful_header =0 
        for col in header :
            h =str (col or "").strip ().lower ()
            if h and not re .fullmatch (r"col_\d+",h ):
                meaningful_header +=1 
        if meaningful_header ==0 and numeric_cells <2 :
            return False 

        joined =" ".join (nonempty )
        has_financial_pattern =bool (
        re .search (r"\b\d{1,3}(?:,\d{3})+\b",joined )
        or re .search (r"\b\d+(?:\.\d+)?%\b",joined )
        or ("$"in joined )
        or re .search (r"\b20\d{2}\b",joined )
        )
        if not has_financial_pattern :
            return False 
        return True 

    def _append_markdown_table_chunks (
    self ,
    chunks :list [dict ],
    table_lines :list [str ],
    h1 :str ,
    h2 :str ,
    source_path :str ,
    table_index :int ,
    )->None :
        header ,body_rows =self ._parse_markdown_table (table_lines )

        # fallback: keep at least retrievable text if parse fails
        if not header and not body_rows :
            raw_content ="\n".join (table_lines ).strip ("\n")
            if raw_content and self .emit_table_raw_chunk :
                chunks .append (
                self ._build_chunk (
                raw_content ,
                h1 ,
                h2 ,
                source_path ,
                is_table =True ,
                table_index =table_index ,
                extra_metadata ={"table_variant":"raw"},
                )
                )
            elif raw_content and self .emit_table_row_facts :
                for idx ,line in enumerate (table_lines [:self .table_max_row_facts ],start =1 ):
                    line_text =self ._normalize_cell (str (line or "").replace ("|"," "))
                    if not line_text :
                        continue 
                    fact =(
                    f"Table text line from {h1} [{h2}] "
                    f"(table {table_index}, line {idx}): {line_text}."
                    )
                    chunks .append (
                    self ._build_chunk (
                    fact ,
                    h1 ,
                    h2 ,
                    source_path ,
                    is_table =True ,
                    table_index =table_index ,
                    extra_metadata ={
                    "table_variant":"row_text_line",
                    "row_id":idx ,
                    "table_name":h2 ,
                    },
                    )
                    )
            return 

        col_count =max (1 ,len (header ))
        header_line =self ._format_markdown_row (header ,col_count )
        separator_line =self ._format_markdown_row (["---"]*col_count ,col_count )

        # 1) raw table (low weight backtracking)
        if self .emit_table_raw_chunk :
            raw_lines =[header_line ,separator_line ]+[
            self ._format_markdown_row (r ,col_count )for r in body_rows 
            ]
            raw_content ="\n".join (raw_lines ).strip ("\n")
            if raw_content :
                chunks .append (
                self ._build_chunk (
                raw_content ,
                h1 ,
                h2 ,
                source_path ,
                is_table =True ,
                table_index =table_index ,
                extra_metadata ={"table_variant":"raw"},
                )
                )

                # 2) table windows (header + several rows)
        if body_rows and self .emit_table_window_chunk :
            for start in range (0 ,len (body_rows ),self .table_window_rows ):
                end =min (start +self .table_window_rows ,len (body_rows ))
                window_rows =body_rows [start :end ]
                window_lines =[header_line ,separator_line ]+[
                self ._format_markdown_row (r ,col_count )for r in window_rows 
                ]
                window_content =(
                f"Table window {table_index} rows {start + 1}-{end}\n"
                +"\n".join (window_lines )
                )
                chunks .append (
                self ._build_chunk (
                window_content ,
                h1 ,
                h2 ,
                source_path ,
                is_table =True ,
                table_index =table_index ,
                extra_metadata ={
                "table_variant":"window",
                "table_row_start":start +1 ,
                "table_row_end":end ,
                },
                )
                )

                # 3) row facts (each row can retrieve natural language)
        if self .emit_table_row_facts and body_rows :
            for idx ,row in enumerate (body_rows [:self .table_max_row_facts ],start =1 ):
                if not self ._row_fact_row_is_valid (header ,row ):
                    continue 
                fact =self ._build_row_fact (
                h1 =h1 ,
                h2 =h2 ,
                table_index =table_index ,
                row_index =idx ,
                header =header ,
                row =row ,
                )
                if not fact :
                    continue 
                chunks .append (
                self ._build_chunk (
                fact ,
                h1 ,
                h2 ,
                source_path ,
                is_table =True ,
                table_index =table_index ,
                extra_metadata ={
                "table_variant":"row_fact",
                "row_id":idx ,
                "table_name":h2 ,
                },
                )
                )

    def split_with_overlap (self ,text ,source_path ):
        lines =text .split ("\n")
        chunks =[]
        buffer_lines =[]
        curr_len ,is_code =0 ,False 
        h1 =os .path .basename (source_path )
        h2 ="Intro"
        table_lines =[]
        in_table =False 
        table_mode =""
        table_index =0 

        def flush_normal (with_overlap :bool )->None :
            nonlocal buffer_lines ,curr_len 
            if not buffer_lines :
                return 

            content ="\n".join (buffer_lines ).strip ("\n")
            if content .strip ():
                chunks .append (
                self ._build_chunk (
                content ,
                h1 ,
                h2 ,
                source_path ,
                is_table =False ,
                )
                )

            if with_overlap and self .overlap_lines >0 :
                if len (buffer_lines )>self .overlap_lines :
                    buffer_lines =buffer_lines [-self .overlap_lines :]
                else :
                    buffer_lines =[]
                curr_len =sum (len (line )for line in buffer_lines )
            else :
                buffer_lines =[]
                curr_len =0 

        def flush_table ()->None :
            nonlocal table_lines ,table_index ,table_mode 
            if not table_lines :
                return 

            content ="\n".join (table_lines ).strip ("\n")
            if content .strip ():
                table_index +=1 
                if table_mode =="markdown":
                    self ._append_markdown_table_chunks (
                    chunks =chunks ,
                    table_lines =table_lines ,
                    h1 =h1 ,
                    h2 =h2 ,
                    source_path =source_path ,
                    table_index =table_index ,
                    )
                else :
                    if self .emit_table_raw_chunk :
                        chunks .append (
                        self ._build_chunk (
                        content ,
                        h1 ,
                        h2 ,
                        source_path ,
                        is_table =True ,
                        table_index =table_index ,
                        extra_metadata ={"table_variant":"raw"},
                        )
                        )
                    elif self .emit_table_row_facts :
                        html_text =re .sub (r"<[^>]+>"," ",content )
                        html_text =re .sub (r"\s+"," ",html_text ).strip ()
                        if html_text :
                            fact =(
                            f"Table text from {h1} [{h2}] (table {table_index}): "
                            f"{html_text[:2000]}."
                            )
                            chunks .append (
                            self ._build_chunk (
                            fact ,
                            h1 ,
                            h2 ,
                            source_path ,
                            is_table =True ,
                            table_index =table_index ,
                            extra_metadata ={
                            "table_variant":"row_text_line",
                            "table_name":h2 ,
                            },
                            )
                            )
            table_lines =[]
            table_mode =""

        for line in lines :
            if line .strip ().startswith ("```"):
                is_code =not is_code 
            if line .startswith ("## "):
                h2 =line .replace ("## ","",1 ).strip ()

            line_mode =self ._table_mode (line )if not is_code else ""

            if in_table :
                if table_mode =="html":
                    table_lines .append (line )
                    if "</table>"in line .lower ():
                        flush_table ()
                        in_table =False 
                    continue 
                if table_mode =="markdown":
                    if self ._is_markdown_table_line (line ):
                        table_lines .append (line )
                        continue 
                    flush_table ()
                    in_table =False 

            if line_mode :
                flush_normal (with_overlap =False )
                in_table =True 
                table_mode =line_mode 
                table_lines =[line ]
                if table_mode =="html"and "</table>"in line .lower ():
                    flush_table ()
                    in_table =False 
                continue 

            buffer_lines .append (line )
            curr_len +=len (line )

            if curr_len >=self .chunk_size and not is_code :
                flush_normal (with_overlap =True )

        if in_table :
            flush_table ()
        if buffer_lines :
            flush_normal (with_overlap =False )
        return chunks 

    def _build_chunk (
    self ,
    content ,
    h1 ,
    h2 ,
    source ,
    is_table =False ,
    table_index =None ,
    extra_metadata =None ,
    ):
        metadata ={
        "h1":h1 ,
        "h2":h2 ,
        "source":source ,
        "is_table":bool (is_table ),
        }
        if is_table and table_index is not None :
            metadata ["table_index"]=int (table_index )
        if isinstance (extra_metadata ,dict )and extra_metadata :
            metadata .update (extra_metadata )

        return {
        "doc_id":self .generate_id (content ),
        "content":content ,
        "metadata":metadata ,
        }
