import time 
import re 
from pathlib import Path 

from config .settings import AppConfig 
from core .llm_call import run_prompt 
from core .state import AgentState ,StepLog 
from llm .prompts .base import PromptContractError 
from llm .prompts .compose_evidence_table import ComposeEvidenceTablePrompt 
from llm .prompts .compose_with_context import ComposeWithContextPrompt 
from nodes .log_utils import clip_text ,preview_citations ,preview_docs ,preview_messages 


_UNCERTAIN_MARKERS =(
"uncertain",
"Unable to determine",
"Insufficient evidence",
"Unable to give verifiable conclusions",
"uncertain",
"cannot determine",
"insufficient evidence",
"not sure",
)


def _is_uncertain_response (text :str )->bool :
    raw =str (text or "").strip ()
    if not raw :
        return True 
    lowered =raw .lower ()
    return any (marker in raw or marker in lowered for marker in _UNCERTAIN_MARKERS )


def _pick_compose_context (context_pool :list [dict ])->list [dict ]:
    top_k =int (AppConfig .RUNTIME_COMPOSE_CONTEXT_TOP_K )
    if top_k <=0 :
        return list (context_pool )
    return list (context_pool [:top_k ])


def _safe_float (value :object ,default :float =0.0 )->float :
    try :
        return float (value )
    except (TypeError ,ValueError ):
        return default 


def _normalize_heading (value :object )->str :
    text =str (value or "").strip ()
    while text .startswith ("#"):
        text =text [1 :]
    return text .strip ()


def _extract_source_and_module (doc :dict )->tuple [str ,str ]:
    metadata =doc .get ("metadata",{})or {}

    source_raw =(
    metadata .get ("source")
    or doc .get ("source")
    or doc .get ("title")
    or doc .get ("id")
    or "Unknown Document"
    )
    source_text =str (source_raw ).strip ()
    source_name =Path (source_text ).name or source_text or "Unknown Document"

    module =(
    _normalize_heading (metadata .get ("h2"))
    or _normalize_heading (metadata .get ("h1"))
    or _normalize_heading (doc .get ("module"))
    or "General"
    )
    route_query =" ".join (str (metadata .get ("__mq_route_query","")).split ()).strip ()
    if route_query :
        module =f"{module} | Route query: {route_query[:120]}"
    return source_name ,module 


def _build_compose_prompt_context (context_pool :list [dict ])->list [dict ]:
    prompt_docs :list [dict ]=[]
    for doc in context_pool :
        source_name ,module =_extract_source_and_module (doc )
        doc_id =str (doc .get ("id","")).strip ()or f"{source_name}::{module}"
        prompt_docs .append (
        {
        "id":doc_id ,
        "title":source_name ,
        "source":source_name ,
        "module":module ,
        "score":_safe_float (doc .get ("score"),0.0 ),
        "content":str (doc .get ("content","")or ""),
        }
        )
    return prompt_docs 


def _context_chars (context_pool :list [dict ])->int :
    chars =0 
    for doc in context_pool :
        chars +=len (str (doc .get ("content","")or ""))
    return chars 


def _build_prompt_state (state :AgentState ,context_pool :list [dict ])->AgentState :
    prompt_state :AgentState =dict (state )
    prompt_state ["context_pool"]=context_pool 
    return prompt_state 


_YEAR_RE =re .compile (r"\b(20\d{2})\b")
_NUMBER_RE =re .compile (r"\$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?|\$?\s*\d+(?:\.\d+)?")
_GROUPED_NUM_RE =re .compile (r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")
_PERIOD_END_RE =re .compile (
r"(?:year\s+ended|ended)\s+([A-Za-z]+\s+\d{1,2},\s*20\d{2})",
flags =re .IGNORECASE ,
)


def _infer_period_end (text :str )->str :
    m =_PERIOD_END_RE .search (str (text or ""))
    if not m :
        return ""
    return " ".join (str (m .group (1 )or "").split ()).strip ()


def _canonical_company_name (text :str )->str :
    lowered =str (text or "").strip ().lower ()
    if not lowered :
        return ""
    for alias ,company in _COMPANY_ALIAS_MAP .items ():
        if alias in lowered :
            return company 
    cleaned =str (text or "").strip ()
    return cleaned 


def _normalize_evidence_year (value :object )->int |None :
    if value is None :
        return None 
    if isinstance (value ,int ):
        if 2000 <=value <=2099 :
            return int (value )
        return None 
    text =str (value or "").strip ()
    m =_YEAR_RE .search (text )
    if not m :
        return None 
    try :
        year =int (m .group (1 ))
    except ValueError :
        return None 
    return year if 2000 <=year <=2099 else None 


def _normalize_evidence_table_rows (
rows :list [dict ],
context_pool :list [dict ],
)->list [dict ]:
    id_to_doc :dict [str ,dict ]={}
    index_to_id :dict [str ,str ]={}
    for doc in context_pool :
        doc_id =str (doc .get ("id","")).strip ()
        if doc_id :
            id_to_doc [doc_id ]=doc 
    for idx ,doc in enumerate (context_pool ,start =1 ):
        doc_id =str (doc .get ("id","")).strip ()
        if doc_id :
            index_to_id [str (idx )]=doc_id 

    normalized :list [dict ]=[]
    seen :set [tuple [str ,str ,int |None ,str ]]=set ()
    for item in rows :
        if not isinstance (item ,dict ):
            continue 

        citation_id =str (
        item .get ("citation_id")
        or item .get ("id")
        or ""
        ).strip ()
        if citation_id in index_to_id :
            citation_id =index_to_id [citation_id ]
        if citation_id and citation_id not in id_to_doc :
            for doc_id in id_to_doc :
                if citation_id .endswith (doc_id )or doc_id .endswith (citation_id ):
                    citation_id =doc_id 
                    break 

        doc =id_to_doc .get (citation_id ,{})
        company =_canonical_company_name (str (item .get ("company","")))
        if not company and doc :
            guessed =sorted (_companies_from_doc_source (doc ))
            if guessed :
                company =guessed [0 ]

        metric =str (item .get ("metric","")).strip ()
        fiscal_year =_normalize_evidence_year (item .get ("fiscal_year"))
        value =str (item .get ("value","")).strip ()
        unit =str (item .get ("unit","")).strip ()or "raw"
        quote =str (item .get ("quote","")).strip ()
        fiscal_period_end =str (item .get ("fiscal_period_end","")).strip ()
        confidence =_safe_float (item .get ("confidence"),0.0 )

        if not quote and doc :
            quote =str (doc .get ("content","")or "").replace ("\n"," ").strip ()[:160 ]
        if not fiscal_period_end :
            fiscal_period_end =_infer_period_end (quote )
        if not fiscal_period_end and doc :
            fiscal_period_end =_infer_period_end (str (doc .get ("content",""))[:400 ])

        if not citation_id and doc :
            citation_id =str (doc .get ("id","")).strip ()

        citation_title =str (item .get ("citation_title","")).strip ()
        if not citation_title and doc :
            citation_title =_source_name_for_doc (doc )

        if not (company or metric or value or citation_id ):
            continue 

        dedup_key =(
        company ,
        metric .lower (),
        fiscal_year ,
        value ,
        )
        if dedup_key in seen :
            continue 
        seen .add (dedup_key )

        normalized .append (
        {
        "company":company ,
        "metric":metric ,
        "fiscal_year":fiscal_year ,
        "fiscal_period_end":fiscal_period_end ,
        "value":value ,
        "unit":unit ,
        "citation_id":citation_id ,
        "citation_title":citation_title ,
        "quote":quote ,
        "confidence":max (0.0 ,min (1.0 ,confidence )),
        }
        )

    normalized .sort (
    key =lambda x :(
    float (x .get ("confidence",0.0 )),
    1 if x .get ("fiscal_year")else 0 ,
    len (str (x .get ("quote",""))),
    ),
    reverse =True ,
    )
    return normalized [:24 ]


def _fallback_citation (doc :dict ,quote_override :str |None =None )->dict |None :
    doc_id =str (doc .get ("id","")).strip ()
    if not doc_id :
        return None 

    title =str (doc .get ("title","")or "Untitled")
    score =float (doc .get ("score",0.0 ))
    content =str (doc .get ("content","")or "").strip ()
    quote =str (quote_override or content ).replace ("\n"," ").strip ()
    if len (quote )>180 :
        quote =quote [:180 ]
    if not quote :
        quote =title 
    return {
    "id":doc_id ,
    "title":title ,
    "score":score ,
    "quote":quote ,
    }


def _source_name_for_doc (doc :dict )->str :
    metadata =doc .get ("metadata",{})or {}
    source_raw =(
    metadata .get ("source")
    or doc .get ("source")
    or doc .get ("title")
    or doc .get ("id")
    or "Unknown Document"
    )
    source_text =str (source_raw ).strip ()
    return Path (source_text ).name or source_text or "Unknown Document"


def _normalize_citations_against_context (
citations :list [dict ],
context_pool :list [dict ],
query :str ,
response :str ="",
)->list [dict ]:
    if not citations or not context_pool :
        return citations 

    id_to_doc :dict [str ,dict ]={}
    for doc in context_pool :
        doc_id =str (doc .get ("id","")).strip ()
        if doc_id :
            id_to_doc [doc_id ]=doc 

    def _source_norm (text :str )->str :
        return Path (str (text or "").strip ()).name .lower ()

    def _target_year (q :str )->str :
        years =_YEAR_RE .findall (str (q or ""))
        return years [-1 ]if years else ""

    def _companies_from_text (text :str )->set [str ]:
        lowered =str (text or "").lower ()
        found :set [str ]=set ()
        for alias ,company in _COMPANY_ALIAS_MAP .items ():
            if alias in lowered :
                found .add (company )
        return found 

    def _find_doc (citation :dict )->dict |None :
        raw_id =str (citation .get ("id","")).strip ()
        id_doc =id_to_doc .get (raw_id )

        # Common mistakes in LLM: appending a serial number before the real ID, or only outputting part of the ID.
        if raw_id and id_doc is None :
            for doc_id ,doc in id_to_doc .items ():
                if raw_id .endswith (doc_id )or doc_id .endswith (raw_id ):
                    id_doc =doc 
                    break 

        title =str (citation .get ("title","")).strip ().lower ()
        quote =str (citation .get ("quote","")).strip ().lower ()
        title_matches :list [dict ]=[]
        for doc in context_pool :
            source_name =_source_name_for_doc (doc ).lower ()
            if title and title not in source_name :
                continue 
            title_matches .append (doc )
        if id_doc is not None :
        # When there is an obvious conflict between id and title, give priority to repositioning with title/quote to avoid quoting the wrong document.
            if title :
                id_source =_source_norm (_source_name_for_doc (id_doc ))
                if title not in id_source and title_matches :
                    if quote :
                        for doc in title_matches :
                            content =str (doc .get ("content","")).lower ()
                            if quote in content :
                                return doc 
                    return title_matches [0 ]
            return id_doc 
        if title_matches and quote :
            for doc in title_matches :
                content =str (doc .get ("content","")).lower ()
                if quote and quote in content :
                    return doc 
        if title_matches :
            return title_matches [0 ]
        return None 

    def _norm_ws (text :str )->str :
        return " ".join (str (text or "").split ()).strip ().lower ()

    def _quote_grounded_in_doc (quote :str ,doc_content :str )->bool :
        q =_norm_ws (quote )
        if not q :
            return False 
        return q in _norm_ws (doc_content )

    target_year =_target_year (str (query or ""))
    response_number_tokens =_response_company_number_tokens (str (response or ""))
    response_numeric_values =_response_company_numeric_values_million (str (response or ""))

    def _evidence_signal (text :str ,company_hints :set [str ])->float :
        lowered =str (text or "").lower ()
        q_terms =_query_terms (query )
        t_terms =set (_TOKEN_PATTERN .findall (lowered ))
        overlap =len (q_terms &t_terms )
        has_number =1.0 if _NUMBER_RE .search (lowered )else 0.0 
        has_year =1.0 if target_year and target_year in lowered else 0.0 
        company_hit =1.0 if (company_hints and (company_hints &_companies_from_text (lowered )))else 0.0 
        metric_hit =1.0 if any (term in lowered for term in ("revenue","sales","cloud","segment"))else 0.0 
        return (overlap *8.0 )+(has_number *3.5 )+(has_year *2.8 )+(company_hit *2.5 )+(metric_hit *1.2 )

    def _numeric_alignment_bonus (text :str ,preferred_values :list [float ])->float :
        if not preferred_values :
            return 0.0 
        actual_values =_extract_numeric_values_million (text )
        if not actual_values :
            return -1.0 
        for actual in actual_values :
            for expected in preferred_values :
                scale =max (abs (expected ),1.0 )
                if abs (actual -expected )/scale <=0.03 :
                    return 8.0 
        return -1.0 

    def _find_better_doc (
    base_doc :dict ,
    company_hints :set [str ],
    preferred_values :list [float ],
    )->dict :
        base_source =_source_norm (_source_name_for_doc (base_doc ))
        base_text =str (base_doc .get ("content",""))
        base_score =_evidence_signal (base_text ,company_hints )+_numeric_alignment_bonus (
        base_text ,
        preferred_values ,
        )
        best_doc =base_doc 
        best_score =base_score 
        for cand in context_pool :
            cand_source =_source_norm (_source_name_for_doc (cand ))
            if base_source and cand_source and cand_source !=base_source :
            # Prioritize better evidence segments within the same source document to avoid cross-document drift.
                continue 
            cand_text =str (cand .get ("content",""))
            cand_score =_evidence_signal (cand_text ,company_hints )+_numeric_alignment_bonus (
            cand_text ,
            preferred_values ,
            )
            if cand_score >best_score :
                best_doc =cand 
                best_score =cand_score 
                # Replace only when it is significantly better to reduce the risk of "over-rewriting".
        if best_score >=base_score +2.5 :
            return best_doc 
        return base_doc 

    normalized :list [dict ]=[]
    seen_ids :set [str ]=set ()
    for citation in citations :
        c =citation or {}
        doc =_find_doc (c )
        if doc is None :
        # Leave it as is and leave it to strict_verify for further judgment.
            normalized .append (c )
            continue 
        company_hints =_companies_from_text (
        " ".join (
        [
        str (c .get ("title","")or ""),
        str (c .get ("id","")or ""),
        str (c .get ("quote","")or ""),
        str (query or ""),
        ]
        )
        )
        preferred_numbers :list [str ]=[]
        preferred_values :list [float ]=[]
        if company_hints :
            for company in company_hints :
                preferred_numbers .extend (response_number_tokens .get (company ,[]))
                preferred_values .extend (response_numeric_values .get (company ,[]))
        doc =_find_better_doc (
        doc ,
        company_hints =company_hints ,
        preferred_values =preferred_values ,
        )
        doc_id =str (doc .get ("id","")).strip ()
        if not doc_id or doc_id in seen_ids :
            continue 
        seen_ids .add (doc_id )
        raw_quote =str (c .get ("quote","")).strip ()
        doc_content =str (doc .get ("content","")or "")
        raw_grounded =bool (raw_quote and _quote_grounded_in_doc (raw_quote ,doc_content ))
        raw_signal =_evidence_signal (raw_quote ,company_hints )if raw_grounded else 0.0 
        if raw_grounded and raw_signal >=6.5 :
            quote =raw_quote 
        else :
            quote =_best_extractive_snippet (
            query =str (query or ""),
            content =doc_content ,
            preferred_numbers =preferred_numbers ,
            preferred_values =preferred_values ,
            )
        normalized .append (
        {
        "id":doc_id ,
        "title":_source_name_for_doc (doc ),
        "score":_safe_float (c .get ("score"),_safe_float (doc .get ("score"),0.0 )),
        "quote":quote ,
        }
        )
    return normalized 


def _ensure_non_empty_citations (response :str ,citations :list [dict ],context_pool :list [dict ])->list [dict ]:
    if citations :
        return citations 
    if _is_uncertain_response (response ):
        return []
    if not context_pool :
        return []
    fallback =_fallback_citation (context_pool [0 ])
    return [fallback ]if fallback else []


_TOKEN_PATTERN =re .compile (r"[a-z0-9]+")
_NUMBER_TOKEN_RE =re .compile (r"\$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?|\$?\s*\d+(?:\.\d+)?")

_COMPANY_ALIAS_MAP :dict [str ,str ]={
"amazon":"Amazon",
"microsoft":"Microsoft",
"msft":"Microsoft",
"alphabet":"Alphabet",
"google":"Alphabet",
}

_REVENUE_HEADING_STRONG_PENALTY_TERMS ={
"cost of sales",
"foreign currency",
"liabilities",
"interest",
"tax",
"effective rate",
"hedging",
"investment",
"allowance",
"cash flow",
"equity",
"debt",
"eps",
}
_REVENUE_HEADING_MEDIUM_PENALTY_TERMS ={
"income",
"expense",
}

def _query_terms (query :str )->set [str ]:
    tokens =_TOKEN_PATTERN .findall (str (query or "").lower ())
    return {tok for tok in tokens if len (tok )>=3 }


def _companies_from_doc_source (doc :dict )->set [str ]:
    metadata =doc .get ("metadata",{})or {}
    lowered =" ".join (
    [
    str (metadata .get ("source","")or ""),
    str (doc .get ("source","")or ""),
    str (doc .get ("title","")or ""),
    str (doc .get ("id","")or ""),
    ]
    ).lower ()
    found :set [str ]=set ()
    for alias ,company in _COMPANY_ALIAS_MAP .items ():
        if alias in lowered :
            found .add (company )
    return found 


def _response_company_number_tokens (response :str )->dict [str ,list [str ]]:
    out :dict [str ,list [str ]]={}
    for sentence in _split_sentences (response ):
        lowered =str (sentence or "").lower ()
        companies :set [str ]=set ()
        for alias ,company in _COMPANY_ALIAS_MAP .items ():
            if alias in lowered :
                companies .add (company )
        if not companies :
            continue 
        parsed_pairs :list [tuple [str ,float ]]=[]
        for m in _NUMBER_TOKEN_RE .finditer (sentence ):
            token =str (m .group (0 )or "").strip ()
            numeric =_parse_numeric_token (token )
            if not token or numeric is None :
                continue 
            if 1900.0 <=numeric <=2100.0 and ","not in token and "."not in token :
                continue 
            window =sentence [max (0 ,m .start ()-24 ):min (len (sentence ),m .end ()+24 )]
            value =_number_to_million (numeric ,window )
            if value <1000.0 :
                continue 
            parsed_pairs .append ((token ,value ))
        if not parsed_pairs :
            continue 
        for company in companies :
            bucket =out .setdefault (company ,[])
            for token ,_ in parsed_pairs :
                if token and token not in bucket :
                    bucket .append (token )
    return out 


def _parse_numeric_token (token :str )->float |None :
    text =str (token or "").strip ().replace ("$","").replace (",","")
    if not text :
        return None 
    try :
        return float (text )
    except ValueError :
        return None 


def _number_to_million (value :float ,window :str )->float :
    lowered =str (window or "").lower ()
    if "billion"in lowered :
        return float (value )*1000.0 
    return float (value )


def _extract_numeric_values_million (text :str )->list [float ]:
    values :list [float ]=[]
    src =str (text or "")
    for match in _NUMBER_TOKEN_RE .finditer (src ):
        token =str (match .group (0 )or "").strip ()
        if not token :
            continue 
        numeric =_parse_numeric_token (token )
        if numeric is None :
            continue 
            # Skip likely year tokens.
        if 1900.0 <=numeric <=2100.0 and ","not in token and "."not in token :
            continue 
        window =src [max (0 ,match .start ()-24 ):min (len (src ),match .end ()+24 )]
        values .append (_number_to_million (numeric ,window ))
    return values 


def _response_company_numeric_values_million (response :str )->dict [str ,list [float ]]:
    out :dict [str ,list [float ]]={}
    for sentence in _split_sentences (response ):
        lowered =str (sentence or "").lower ()
        companies :set [str ]=set ()
        for alias ,company in _COMPANY_ALIAS_MAP .items ():
            if alias in lowered :
                companies .add (company )
        if not companies :
            continue 
        values =_extract_numeric_values_million (sentence )
        if not values :
            continue 
        for company in companies :
            bucket =out .setdefault (company ,[])
            for value in values :
            # Prefer materially meaningful financial magnitudes; drop tiny/date-like residues.
                if value <1000.0 :
                    continue 
                if all (abs (value -existing )/max (abs (existing ),1.0 )>0.01 for existing in bucket ):
                    bucket .append (value )
    return out 


def _ensure_compose_route_coverage (
compose_context_pool :list [dict ],
full_context_pool :list [dict ],
phase1_query_routes :list [dict ],
)->tuple [list [dict ],list [int ]]:
    if len (phase1_query_routes )<2 :
        return compose_context_pool ,[]

    composed =list (compose_context_pool )
    existing_ids ={str (d .get ("id","")).strip ()for d in composed }
    full_doc_map ={
    str (doc .get ("id","")).strip ():doc 
    for doc in full_context_pool 
    if str (doc .get ("id","")).strip ()
    }

    missing_routes :list [int ]=[]
    per_route_top_k =max (1 ,int (AppConfig .RETRIEVAL_MULTI_QUERY_CONTEXT_TOP_K ))
    for route in phase1_query_routes :
        route_idx =int (route .get ("route_index",0 ))
        route_candidates =list (route .get ("candidates",[])or [])
        route_ids =[
        str (c .get ("id","")).strip ()
        for c in route_candidates 
        if str (c .get ("id","")).strip ()
        ]
        if not route_ids :
            continue 
        expected_k =min (per_route_top_k ,len (route_ids ))
        current_count =sum (1 for doc_id in route_ids if doc_id in existing_ids )
        if current_count >=expected_k :
            continue 
        if current_count <=0 :
            missing_routes .append (route_idx )
        for candidate in route_candidates :
            candidate_id =str (candidate .get ("id","")).strip ()
            if not candidate_id :
                continue 
            doc =full_doc_map .get (candidate_id )or candidate 
            if candidate_id in existing_ids :
                continue 
            composed .append (doc )
            existing_ids .add (candidate_id )
            current_count +=1 
            if current_count >=expected_k :
                break 

    return composed ,missing_routes 


def _route_doc_quality (route_query :str ,doc :dict )->float :
    q =str (route_query or "").lower ()
    content =str (doc .get ("content","")or "")
    c =content .lower ()
    heading =str ((doc .get ("metadata",{})or {}).get ("h2","")or "").lower ()

    q_terms =_query_terms (q )
    c_terms =set (_TOKEN_PATTERN .findall (c ))
    overlap =float (len (q_terms &c_terms ))/float (max (1 ,len (q_terms )))

    years =set (_YEAR_RE .findall (q ))
    year_hit =1.0 if (years and any (y in c for y in years ))else 0.0 
    numeric_density =min (1.0 ,float (len (_GROUPED_NUM_RE .findall (content )))/8.0 )

    anchor_hits =0.0 
    if "aws"in q and "aws"in c :
        anchor_hits +=1.0 
    if "google cloud"in q and "google cloud"in c :
        anchor_hits +=1.0 
    if ("azure"in q or "microsoft cloud"in q )and ("azure"in c or "microsoft cloud"in c ):
        anchor_hits +=1.0 
    if any (t in q for t in ("revenue","sales","segment"))and (
    "revenue"in c or "sales"in c or "net sales"in c 
    ):
        anchor_hits +=1.0 
    if "cash and cash equivalents"in q and "cash and cash equivalents"in c :
        anchor_hits +=1.0 
    anchor_score =min (1.0 ,anchor_hits /3.0 )

    penalty =0.0 
    asks_revenue =any (t in q for t in ("revenue","sales","segment"))
    heading_is_revenue =("revenue"in heading )or ("net sales"in heading )or ("cloud"in heading )
    if "cost of sales"in heading :
        heading_is_revenue =False 
    if asks_revenue and not heading_is_revenue :
        if any (term in heading for term in _REVENUE_HEADING_STRONG_PENALTY_TERMS ):
            penalty =0.34 
        elif any (term in heading for term in _REVENUE_HEADING_MEDIUM_PENALTY_TERMS ):
            penalty =0.22 

    balance_boost =0.0 
    asks_balance =any (t in q for t in ("balance","year-end","as of","end of"))
    asks_cash_eq =("cash and cash equivalents"in q )or ("cash equivalents"in q )
    if asks_balance :
        if any (term in heading for term in ("balance sheet","balance sheets","assets")):
            balance_boost +=0.14 
        if "consolidated balance sheets"in c :
            balance_boost +=0.12 
    if asks_cash_eq :
        if "cash and cash equivalents"in c :
            balance_boost +=0.14 
        if "cash, cash equivalents, and short term marketable securities"in c :
            penalty +=0.08 
        if "cash, cash equivalents, and marketable securities"in c :
            penalty +=0.06 
    if asks_balance and any (term in heading for term in ("stockholders","comprehensive income","operating activities")):
        penalty +=0.12 

    return (
    (overlap *0.22 )
    +(year_hit *0.08 )
    +(numeric_density *0.14 )
    +(anchor_score *0.16 )
    +balance_boost 
    -penalty 
    )


def _prioritize_route_evidence_docs (
compose_context_pool :list [dict ],
phase1_query_routes :list [dict ],
)->list [dict ]:
    if len (phase1_query_routes )<2 :
        return compose_context_pool 

    doc_map :dict [str ,dict ]={}
    for doc in compose_context_pool :
        doc_id =str (doc .get ("id","")).strip ()
        if doc_id :
            doc_map [doc_id ]=doc 

    selected_buckets :list [list [dict ]]=[]
    for route in phase1_query_routes :
        route_query =str (route .get ("query","")or "").strip ()
        candidates =list (route .get ("candidates",[])or [])
        if not candidates :
            continue 
        scored :list [tuple [float ,dict ]]=[]
        for cand in candidates :
            doc_id =str (cand .get ("id","")).strip ()
            doc =doc_map .get (doc_id )or cand 
            if not doc_id :
                continue 
            scored .append ((_route_doc_quality (route_query =route_query ,doc =doc ),doc ))
        if not scored :
            continue 
        scored .sort (key =lambda x :x [0 ],reverse =True )
        selected_buckets .append ([doc for _ ,doc in scored [:2 ]])

    prioritized :list [dict ]=[]
    seen :set [str ]=set ()
    level =0 
    while True :
        progressed =False 
        for bucket in selected_buckets :
            if level >=len (bucket ):
                continue 
            progressed =True 
            doc =bucket [level ]
            doc_id =str (doc .get ("id","")).strip ()
            if not doc_id or doc_id in seen :
                continue 
            seen .add (doc_id )
            prioritized .append (doc )
        if not progressed :
            break 
        level +=1 

    for doc in compose_context_pool :
        doc_id =str (doc .get ("id","")).strip ()
        if not doc_id or doc_id in seen :
            continue 
        prioritized .append (doc )
    return prioritized 


def _numeric_alignment_bonus_for_doc (content :str ,preferred_values :list [float ])->float :
    if not preferred_values :
        return 0.0 
    actual_values =_extract_numeric_values_million (content )
    if not actual_values :
        return -0.5 
    for actual in actual_values :
        for expected in preferred_values :
            scale =max (abs (expected ),1.0 )
            if abs (actual -expected )/scale <=0.03 :
                return 5.0 
    return -0.5 


def _build_route_aligned_citations (
query :str ,
response :str ,
compose_context_pool :list [dict ],
phase1_query_routes :list [dict ],
)->list [dict ]:
    if len (phase1_query_routes )<2 :
        return []

    response_tokens =_response_company_number_tokens (response )
    response_values =_response_company_numeric_values_million (response )

    doc_map :dict [str ,dict ]={}
    for doc in compose_context_pool :
        doc_id =str (doc .get ("id","")).strip ()
        if doc_id :
            doc_map [doc_id ]=doc 

    out :list [dict ]=[]
    seen_ids :set [str ]=set ()
    for route in phase1_query_routes :
        route_query =str (route .get ("query","")or "").strip ()
        candidates =list (route .get ("candidates",[])or [])
        if not route_query or not candidates :
            continue 

        companies :set [str ]=set ()
        lowered_query =route_query .lower ()
        for alias ,company in _COMPANY_ALIAS_MAP .items ():
            if alias in lowered_query :
                companies .add (company )

        preferred_numbers :list [str ]=[]
        preferred_values :list [float ]=[]
        for company in companies :
            preferred_numbers .extend (response_tokens .get (company ,[]))
            preferred_values .extend (response_values .get (company ,[]))

        best_doc :dict |None =None 
        best_score =-1e9 
        for cand in candidates :
            doc_id =str (cand .get ("id","")).strip ()
            doc =doc_map .get (doc_id )or cand 
            content =str (doc .get ("content","")or "")
            score =_route_doc_quality (route_query =route_query ,doc =doc )
            score +=_numeric_alignment_bonus_for_doc (content =content ,preferred_values =preferred_values )
            if score >best_score :
                best_score =score 
                best_doc =doc 

        if not best_doc :
            continue 
        doc_id =str (best_doc .get ("id","")).strip ()
        if not doc_id or doc_id in seen_ids :
            continue 
        seen_ids .add (doc_id )
        quote =_best_extractive_snippet (
        query =route_query ,
        content =str (best_doc .get ("content","")or ""),
        preferred_numbers =preferred_numbers ,
        preferred_values =preferred_values ,
        )
        out .append (
        {
        "id":doc_id ,
        "title":_source_name_for_doc (best_doc ),
        "score":_safe_float (best_doc .get ("score"),0.0 ),
        "quote":quote ,
        }
        )
    return out 


def _split_sentences (text :str )->list [str ]:
    parts =re .split (r"(?<=[.!?。！？])\s+|\n+",str (text or ""))
    return [p .strip ()for p in parts if p and p .strip ()]


def _best_extractive_snippet (
query :str ,
content :str ,
preferred_numbers :list [str ]|None =None ,
preferred_values :list [float ]|None =None ,
)->str :
    source =str (content or "")
    if not source .strip ():
        return ""

    target_year =""
    years =_YEAR_RE .findall (str (query or ""))
    if years :
        target_year =years [-1 ]
    q_terms =_query_terms (query )

    candidates :list [str ]=[]
    if target_year :
        cursor =0 
        while True :
            idx =source .find (target_year ,cursor )
            if idx <0 :
                break 
            start =max (0 ,idx -96 )
            end =min (len (source ),idx +240 )
            window =source [start :end ]
            candidates .append (window )
            cursor =idx +len (target_year )
            if len (candidates )>=6 :
                break 

    preferred =[str (x or "").strip ()for x in (preferred_numbers or [])if str (x or "").strip ()]
    if preferred :
        lowered_source =source .lower ()
        for token in preferred :
            for needle in {token .lower (),token .lower ().replace (",","")}:
                if not needle :
                    continue 
                cursor =0 
                while True :
                    idx =lowered_source .find (needle ,cursor )
                    if idx <0 :
                        break 
                    start =max (0 ,idx -60 )
                    end =min (len (source ),idx +max (150 ,len (needle )+48 ))
                    candidates .append (source [start :end ])
                    cursor =idx +len (needle )
                    if len (candidates )>=16 :
                        break 
                if len (candidates )>=16 :
                    break 
            if len (candidates )>=16 :
                break 

    preferred_numeric_values =list (preferred_values or [])
    if preferred_numeric_values :
        for m in _NUMBER_TOKEN_RE .finditer (source ):
            token =str (m .group (0 )or "").strip ()
            numeric =_parse_numeric_token (token )
            if numeric is None :
                continue 
            window =source [max (0 ,m .start ()-24 ):min (len (source ),m .end ()+24 )]
            value =_number_to_million (numeric ,window )
            matched =False 
            for expected in preferred_numeric_values :
                scale =max (abs (expected ),1.0 )
                if abs (value -expected )/scale <=0.03 :
                    matched =True 
                    break 
            if not matched :
                continue 
            start =max (0 ,m .start ()-60 )
            end =min (len (source ),m .end ()+160 )
            candidates .append (source [start :end ])
            if len (candidates )>=24 :
                break 

    sentences =_split_sentences (source )
    candidates .extend (sentences [:16 ])
    if not candidates :
        return source .strip ()[:260 ]

    best_text =candidates [0 ]
    best_score =-1.0 
    for cand in candidates :
        lowered =cand .lower ()
        terms =set (_TOKEN_PATTERN .findall (lowered ))
        overlap =len (q_terms &terms )
        has_number =1 if _NUMBER_RE .search (cand )else 0 
        has_year =1 if target_year and target_year in cand else 0 
        metric_hit =1 if any (k in lowered for k in ("revenue","sales","cloud","segment","aws","azure","google"))else 0 
        preferred_hit =0 
        cand_compact =lowered .replace (",","")
        for token in preferred :
            t =token .lower ()
            if not t :
                continue 
            if t in lowered or t .replace (",","")in cand_compact :
                preferred_hit =1 
                break 
        preferred_value_hit =0 
        if preferred_numeric_values :
            for m in _NUMBER_TOKEN_RE .finditer (cand ):
                token =str (m .group (0 )or "").strip ()
                numeric =_parse_numeric_token (token )
                if numeric is None :
                    continue 
                window =cand [max (0 ,m .start ()-24 ):min (len (cand ),m .end ()+24 )]
                value =_number_to_million (numeric ,window )
                for expected in preferred_numeric_values :
                    scale =max (abs (expected ),1.0 )
                    if abs (value -expected )/scale <=0.03 :
                        preferred_value_hit =1 
                        break 
                if preferred_value_hit :
                    break 
        score =(
        (overlap *12.0 )
        +(has_number *6.0 )
        +(has_year *5.0 )
        +(metric_hit *3.0 )
        +(preferred_hit *12.0 )
        +(preferred_value_hit *40.0 )
        )
        if score >best_score :
            best_score =score 
            best_text =cand 

    return " ".join (str (best_text or "").split ())[:260 ]


def _select_repair_anchor_doc (context_pool :list [dict ],citations :list [dict ])->dict |None :
    if not context_pool :
        return None 

    citation_id =""
    if citations :
        citation_id =str ((citations [0 ]or {}).get ("id","")).strip ()
    if citation_id :
        for doc in context_pool :
            if str (doc .get ("id","")).strip ()==citation_id :
                return doc 
    return context_pool [0 ]


def _should_apply_repair_extractive_fallback (
state :AgentState ,
response :str ,
citations :list [dict ],
context_pool :list [dict ],
)->bool :
    if not bool (state .get ("repair_mode",False )):
        return False 
    if not bool (AppConfig .RUNTIME_REPAIR_EXTRACTIVE_FALLBACK ):
        return False 
    if not context_pool :
        return False 
    if len (state .get ("retrieval_queries",[])or [])>1 :
    # Multi-subquery problems require cross-document synthesis, and single-document extraction will significantly degrade the answer.
        return False 

    failure_type =str (state .get ("failure_type","")).strip ().upper ()
    if failure_type in {"LOGICAL_ERROR","CITATION_MISMATCH","FORMAT_ERROR"}:
        return True 

    return _is_uncertain_response (response )or (not citations )


def compose_with_context_node (state :AgentState )->AgentState :
    state .setdefault ("repair_mode",False )
    state .setdefault ("strict_reason","")
    state .setdefault ("previous_response",state .get ("response",""))
    state .setdefault ("evidence_table",[])
    state .setdefault ("route_facts",[])
    state .setdefault ("route_fact_coverage",{})
    state .setdefault ("resolved_query",str (state .get ("query","")or ""))
    state .setdefault ("retrieval_query",str (state .get ("resolved_query")or state .get ("query","")or ""))
    if not isinstance (state .get ("retrieval_queries"),list )or not (state .get ("retrieval_queries")or []):
        fallback_retrieval_query =str (state .get ("retrieval_query","")or "").strip ()
        state ["retrieval_queries"]=[fallback_retrieval_query ]if fallback_retrieval_query else []

    full_context_pool =state .get ("context_pool",[])or []
    phase1_query_routes =state .get ("phase1_query_routes",[])or []
    phase2_query_routes =state .get ("phase2_query_routes",[])or []
    context_source =str (state .get ("context_source","")or "").strip ().lower ()
    if context_source =="phase2":
        active_query_routes =phase2_query_routes or phase1_query_routes 
    else :
        active_query_routes =phase1_query_routes or phase2_query_routes 
    if bool (AppConfig .RUNTIME_COMPOSE_INCLUDE_ALL_RETRIEVED_CONTEXT ):
        compose_context_pool =list (full_context_pool )
    else :
        compose_context_pool =_pick_compose_context (full_context_pool )
    compose_context_pool ,missing_query_routes =_ensure_compose_route_coverage (
    compose_context_pool =compose_context_pool ,
    full_context_pool =full_context_pool ,
    phase1_query_routes =active_query_routes ,
    )
    compose_context_pool =_prioritize_route_evidence_docs (
    compose_context_pool =compose_context_pool ,
    phase1_query_routes =active_query_routes ,
    )
    prompt_context_pool =_build_compose_prompt_context (compose_context_pool )

    evidence_table_error =""
    evidence_table :list [dict ]=[]
    if bool (AppConfig .RUNTIME_ENABLE_EVIDENCE_TABLE ):
        try :
            evidence_state =_build_prompt_state (state ,prompt_context_pool )
            evidence_out =run_prompt (ComposeEvidenceTablePrompt ,evidence_state )
            evidence_rows =evidence_out .get ("evidence_table",[])or []
            evidence_table =_normalize_evidence_table_rows (
            rows =evidence_rows ,
            context_pool =compose_context_pool ,
            )
        except PromptContractError as e :
            evidence_table_error =f"{type(e).__name__}: {e}"
        except Exception as e :# pragma: no cover
            evidence_table_error =f"{type(e).__name__}: {e}"
    else :
        evidence_table_error ="disabled_by_config:RUNTIME_ENABLE_EVIDENCE_TABLE=false"

    state ["evidence_table"]=evidence_table 
    prompt_state =_build_prompt_state (state ,prompt_context_pool )
    prompt_state ["evidence_table"]=evidence_table 
    out =run_prompt (ComposeWithContextPrompt ,prompt_state )

    response =str (out .get ("response","")).strip ()
    citations =out .get ("citations",[])or []
    forced_retry =False 
    repair_extractive_applied =False 

    # If the model still outputs "uncertain" despite existing evidence, a forced extraction retry is triggered.
    if (
    bool (AppConfig .RUNTIME_FORCE_ANSWER_ON_EVIDENCE )
    and compose_context_pool 
    and _is_uncertain_response (response )
    ):
        forced_retry =True 
        retry_state =_build_prompt_state (state ,prompt_context_pool )
        retry_state ["repair_mode"]=True 
        retry_state ["strict_reason"]=(
        "The previous answer was uncertain. Evidence exists in context_pool. "
        "Extract the best-supported direct answer and provide at least one citation."
        )
        retry_state ["previous_response"]=response 
        retry_out =run_prompt (ComposeWithContextPrompt ,retry_state )
        retry_response =str (retry_out .get ("response","")).strip ()
        retry_citations =retry_out .get ("citations",[])or []
        if retry_response :
            response =retry_response 
            citations =retry_citations 

            # In the repair phase, priority is given to converting to extractive answers to avoid logical expansion errors from happening again.
    if _should_apply_repair_extractive_fallback (
    state =state ,
    response =response ,
    citations =citations ,
    context_pool =compose_context_pool ,
    ):
        anchor_doc =_select_repair_anchor_doc (compose_context_pool ,citations )
        if anchor_doc :
            snippet =_best_extractive_snippet (
            query =str (state .get ("query","")),
            content =str (anchor_doc .get ("content","")),
            )
            fallback_citation =_fallback_citation (anchor_doc ,quote_override =snippet )
            if fallback_citation :
                response =snippet or response 
                citations =[fallback_citation ]
                repair_extractive_applied =True 

    citations =_normalize_citations_against_context (
    citations =citations ,
    context_pool =compose_context_pool ,
    query =str (state .get ("resolved_query")or state .get ("query","")or ""),
    response =response ,
    )
    route_aligned_citations =_build_route_aligned_citations (
    query =str (state .get ("resolved_query")or state .get ("query","")or ""),
    response =response ,
    compose_context_pool =compose_context_pool ,
    phase1_query_routes =active_query_routes ,
    )
    if route_aligned_citations :
        citations =route_aligned_citations 

    state ["previous_response"]=state .get ("response","")
    state ["response"]=response 
    state ["citations"]=_ensure_non_empty_citations (response ,citations ,compose_context_pool )
    state ["response_revision"]=state .get ("response_revision",0 )+1 

    state .setdefault ("steps_log",[]).append (
    StepLog (
    node ="compose_with_context",
    info ={
    "state":{
    "query_preview":clip_text (state .get ("query",""),180 ),
    "repair_mode":bool (state .get ("repair_mode",False )),
    "context_pool_count":len (full_context_pool ),
    "compose_context_count":len (compose_context_pool ),
    "compose_context_chars":_context_chars (compose_context_pool ),
    "compose_prompt_context_chars":_context_chars (prompt_context_pool ),
    "forced_retry":forced_retry ,
    "query_route_count":len (active_query_routes ),
    "missing_query_routes_before_patch":missing_query_routes ,
    "compose_include_all_retrieved_context":bool (
    AppConfig .RUNTIME_COMPOSE_INCLUDE_ALL_RETRIEVED_CONTEXT 
    ),
    "evidence_table_count":len (state .get ("evidence_table",[])or []),
    "evidence_table_error_preview":clip_text (evidence_table_error ,200 ),
    "repair_extractive_applied":repair_extractive_applied ,
    "route_aligned_citations_count":len (route_aligned_citations ),
    "response_revision":state .get ("response_revision",0 ),
    },
    "llm_input":{
    "short_term_memory_preview":clip_text (state .get ("short_term_memory",""),160 ),
    "recent_messages_preview":preview_messages (state .get ("recent_messages",[])),
    "long_term_memory_preview":clip_text (state .get ("long_term_memory",""),160 ),
    "strict_reason_preview":clip_text (state .get ("strict_reason",""),160 ),
    "context_pool_preview":preview_docs (prompt_context_pool ),
    "evidence_table_preview":state .get ("evidence_table",[])[:4 ],
    },
    "llm_output":{
    "response_preview":clip_text (state .get ("response",""),220 ),
    "citations_preview":preview_citations (state .get ("citations",[])),
    },
    },
    timestamp =time .time (),
    )
    )
    return state 
