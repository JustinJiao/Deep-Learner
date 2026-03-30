import time 
import re 
from typing import Any 
from pathlib import Path 

from config .settings import AppConfig 
from core .llm_call import run_prompt 
from core .state import AgentState ,StepLog 
from llm .prompts .strict_verify import StrictVerifyPrompt 
from nodes .log_utils import clip_text ,preview_docs 

ALLOWED_FAILURE_TYPES ={
"INSUFFICIENT_EVIDENCE",
"LOGICAL_ERROR",
"CITATION_MISMATCH",
"FORMAT_ERROR",
}

_TRIGGER_TO_FAILURE_TYPE ={
"citation_missing":"CITATION_MISMATCH",
"citation_fabricated":"CITATION_MISMATCH",
"citation_not_in_context":"CITATION_MISMATCH",
"unsupported_claim":"LOGICAL_ERROR",
"logic_contradiction":"LOGICAL_ERROR",
"citation_score_too_low":"CITATION_MISMATCH",
"hallucination_score_too_low":"INSUFFICIENT_EVIDENCE",
"logic_score_too_low":"LOGICAL_ERROR",
"total_score_below_threshold":"LOGICAL_ERROR",
"insufficient_evidence_missing_company":"INSUFFICIENT_EVIDENCE",
"numeric_year_value_mismatch":"LOGICAL_ERROR",
"numeric_claim_not_in_citation":"CITATION_MISMATCH",
}

_MISSING_EVIDENCE_PATTERN =re .compile (
r"missing\s+explicit\s+evidence\s+for\s*:\s*([^\n\.]+)",
flags =re .IGNORECASE ,
)

_COMPANY_ALIAS_MAP :dict [str ,str ]={
"amazon":"Amazon",
"microsoft":"Microsoft",
"msft":"Microsoft",
"alphabet":"Alphabet",
"google":"Alphabet",
}

_COMPANY_ALIASES :dict [str ,tuple [str ,...]]={
"Amazon":("amazon","aws"),
"Microsoft":("microsoft","msft"),
"Alphabet":("alphabet","google","google cloud"),
}

_YEAR_RE =re .compile (r"\b(20\d{2})\b")
_NUMBER_RE =re .compile (r"\$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?|\$?\s*\d+(?:\.\d+)?")
_SENTENCE_SPLIT_RE =re .compile (r"(?<=[.!?。！？])\s+|\n+")


def _target_year (query :str ,resolved_query :str )->int |None :
    for text in (resolved_query ,query ):
        years =[int (y )for y in _YEAR_RE .findall (str (text or ""))]
        if years :
            return max (years )
    return None 


def _parse_number (raw :str )->float |None :
    text =str (raw or "").strip ().replace ("$","").replace (",","")
    if not text :
        return None 
    try :
        return float (text )
    except ValueError :
        return None 


def _number_to_million (value :float ,unit_hint :str )->float :
    lowered =str (unit_hint or "").lower ()
    if "billion"in lowered :
        return float (value )*1000.0 
    if "million"in lowered :
        return float (value )
    return float (value )


def _is_financial_number_token (token :str ,unit_hint_window :str )->bool :
    t =str (token or "")
    window =str (unit_hint_window or "").lower ()
    if "$"in t :
        return True 
    if ","in t :
        return True 
    if "."in t :
        return True 
    if "million"in window or "billion"in window :
        return True 
    return False 


def _extract_response_company_values_million (response :str )->dict [str ,list [float ]]:
    mentions :dict [str ,list [float ]]={}
    for sentence in _SENTENCE_SPLIT_RE .split (str (response or "")):
        sent =str (sentence or "").strip ()
        if not sent :
            continue 
        lowered =sent .lower ()
        companies =[
        company 
        for company ,aliases in _COMPANY_ALIASES .items ()
        if any (alias in lowered for alias in aliases )
        ]
        if not companies :
            continue 

        for match in _NUMBER_RE .finditer (sent ):
            token =str (match .group (0 )or "").strip ()
            if not token :
                continue 

            end =int (match .end ())
            if end <len (sent )and sent [end :end +1 ]=="%":
                continue 

            numeric =_parse_number (token )
            if numeric is None :
                continue 

                # Skip likely year mentions.
            if 1900.0 <=numeric <=2100.0 and ","not in token and "."not in token :
                continue 

            window =sent [max (0 ,match .start ()-24 ):min (len (sent ),match .end ()+24 )].lower ()
            if not _is_financial_number_token (token =token ,unit_hint_window =window ):
                continue 
            value_million =_number_to_million (numeric ,unit_hint =window )
            for company in companies :
                mentions .setdefault (company ,[]).append (value_million )
    return mentions 


def _extract_numeric_values_million (text :str )->list [float ]:
    values :list [float ]=[]
    for match in _NUMBER_RE .finditer (str (text or "")):
        token =str (match .group (0 )or "").strip ()
        if not token :
            continue 

        end =int (match .end ())
        src =str (text or "")
        if end <len (src )and src [end :end +1 ]=="%":
            continue 

        numeric =_parse_number (token )
        if numeric is None :
            continue 

            # Skip likely year mentions.
        if 1900.0 <=numeric <=2100.0 and ","not in token and "."not in token :
            continue 

        window =src [max (0 ,match .start ()-24 ):min (len (src ),match .end ()+24 )].lower ()
        if not _is_financial_number_token (token =token ,unit_hint_window =window ):
            continue 
        values .append (_number_to_million (numeric ,unit_hint =window ))
    return values 


def _extract_citation_company_numeric_support (
citations :list [dict ],
context_pool :list [dict ],
)->dict [str ,dict [str ,Any ]]:
    by_company :dict [str ,dict [str ,Any ]]={}
    id_to_doc :dict [str ,dict ]={}
    for doc in context_pool :
        doc_id =str (doc .get ("id","")).strip ()
        if doc_id :
            id_to_doc [doc_id ]=doc 

    for citation in citations :
        c =citation or {}
        quote =str (c .get ("quote","")or "")
        title =str (c .get ("title","")or "")
        citation_id =str (c .get ("id","")or "")
        doc_content =str ((id_to_doc .get (citation_id ,{})or {}).get ("content","")or "")
        local_support_text =quote 
        if quote and doc_content :
            idx =doc_content .lower ().find (str (quote ).lower ())
            if idx >=0 :
                start =max (0 ,idx -240 )
                end =min (len (doc_content ),idx +len (quote )+240 )
                local_support_text =f"{quote}\n{doc_content[start:end]}"
            else :
                local_support_text =quote 

        companies =_companies_from_text (" ".join ([title ,citation_id ,local_support_text [:360 ]]))
        if not companies :
            continue 

        values =_extract_numeric_values_million (local_support_text )
        years ={int (y )for y in _YEAR_RE .findall (local_support_text )}
        if not values :
            continue 

        for company in companies :
            entry =by_company .setdefault (company ,{"values":[],"years":set ()})
            entry ["values"].extend (values )
            entry ["years"].update (years )
    return by_company 


def _detect_numeric_claim_citation_mismatch (
query :str ,
resolved_query :str ,
response :str ,
citations :list [dict ],
context_pool :list [dict ],
)->list [str ]:
    target_year =_target_year (query =query ,resolved_query =resolved_query )
    response_by_company =_extract_response_company_values_million (response =response )
    if not response_by_company :
        return []

    citation_by_company =_extract_citation_company_numeric_support (
    citations =citations ,
    context_pool =context_pool ,
    )
    mismatches :list [str ]=[]
    for company ,response_values in response_by_company .items ():
        support =citation_by_company .get (company ,{})
        citation_values =list (support .get ("values",[])or [])
        citation_years ={int (y )for y in (support .get ("years",set ())or set ())if 2000 <=int (y )<=2099 }
        if not citation_values :
            response_preview =", ".join (f"{v:.3f}"for v in response_values [:3 ])
            mismatches .append (
            f"{company} response_values=[{response_preview}] but citation_has_no_numeric_value"
            )
            continue 

        if target_year is not None and citation_years and target_year not in citation_years :
            mismatches .append (
            f"{company} citation_years={sorted(citation_years)} missing_target_year={target_year}"
            )
            continue 

        matched =any (
        _value_close (actual ,expected )
        for actual in response_values 
        for expected in citation_values 
        )
        if matched :
            continue 

        response_preview =", ".join (f"{v:.3f}"for v in response_values [:3 ])
        citation_preview =", ".join (f"{v:.3f}"for v in citation_values [:3 ])
        mismatches .append (
        f"{company} response_values=[{response_preview}] "
        f"but citation_values=[{citation_preview}]"
        )
    return mismatches 


def _extract_evidence_expected_values_million (
evidence_table :list [dict ],
target_year :int |None ,
)->dict [str ,list [float ]]:
    expected :dict [str ,list [float ]]={}
    for row in evidence_table :
        if not isinstance (row ,dict ):
            continue 
        company =_canonical_company (str (row .get ("company","")))
        if not company :
            continue 

        row_year =_normalize_evidence_year (row .get ("fiscal_year"))
        if target_year is not None and row_year !=target_year :
            continue 

        numeric =_parse_number (str (row .get ("value","")))
        if numeric is None :
            continue 
        value_million =_number_to_million (numeric ,unit_hint =str (row .get ("unit","")))
        expected .setdefault (company ,[]).append (value_million )
    return expected 


def _normalize_evidence_year (value :object )->int |None :
    if value is None :
        return None 
    if isinstance (value ,int ):
        return value if 2000 <=value <=2099 else None 
    years =_YEAR_RE .findall (str (value or ""))
    if not years :
        return None 
    try :
        year =int (years [-1 ])
    except ValueError :
        return None 
    return year if 2000 <=year <=2099 else None 


def _canonical_company (text :str )->str :
    lowered =str (text or "").strip ().lower ()
    if not lowered :
        return ""
    for alias ,company in _COMPANY_ALIAS_MAP .items ():
        if alias in lowered :
            return company 
    for company ,aliases in _COMPANY_ALIASES .items ():
        if any (alias in lowered for alias in aliases ):
            return company 
    return str (text or "").strip ()


def _value_close (a :float ,b :float )->bool :
    scale =max (abs (b ),1.0 )
    return abs (a -b )/scale <=0.02 


def _numeric_alignment_summary (
query :str ,
resolved_query :str ,
response :str ,
evidence_table :list [dict ],
)->dict [str ,Any ]:
    year =_target_year (query =query ,resolved_query =resolved_query )
    expected_by_company =_extract_evidence_expected_values_million (
    evidence_table =evidence_table ,
    target_year =year ,
    )
    response_by_company =_extract_response_company_values_million (response =response )

    mismatches :list [str ]=[]
    matched_companies :set [str ]=set ()
    missing_in_response :set [str ]=set ()
    for company ,expected_values in expected_by_company .items ():
        actual_values =response_by_company .get (company ,[])
        if not actual_values :
            missing_in_response .add (company )
            continue 

        matched =any (
        _value_close (actual ,expected )
        for actual in actual_values 
        for expected in expected_values 
        )
        if matched :
            matched_companies .add (company )
            continue 

        expected_preview =", ".join (f"{v:.3f}"for v in expected_values [:3 ])
        actual_preview =", ".join (f"{v:.3f}"for v in actual_values [:3 ])
        mismatches .append (
        f"{company}@{year} expected_million=[{expected_preview}] "
        f"but_response=[{actual_preview}]"
        )

    strong_alignment =bool (
    expected_by_company 
    and not mismatches 
    and not missing_in_response 
    and matched_companies ==set (expected_by_company .keys ())
    )
    return {
    "year":year ,
    "expected_by_company":expected_by_company ,
    "matched_companies":matched_companies ,
    "missing_in_response":missing_in_response ,
    "mismatches":mismatches ,
    "strong_alignment":strong_alignment ,
    }


def _clamp_score_0_5 (value :object ,default :float =0.0 )->float :
    try :
        score =float (value )
    except (TypeError ,ValueError ):
        return default 
    return max (0.0 ,min (5.0 ,score ))


def _clamp_confidence (value :object ,default :float =0.0 )->float :
    try :
        score =float (value )
    except (TypeError ,ValueError ):
        return default 
    return max (0.0 ,min (1.0 ,score ))


def _as_bool (value :object ,default :bool =False )->bool :
    if isinstance (value ,bool ):
        return value 
    if isinstance (value ,(int ,float )):
        return bool (value )
    text =str (value or "").strip ().lower ()
    if text in {"true","1","yes","on"}:
        return True 
    if text in {"false","0","no","off"}:
        return False 
    return default 


def _pick_verify_context (context_pool :list [dict ])->list [dict ]:
    top_k =int (AppConfig .RUNTIME_VERIFY_CONTEXT_TOP_K )
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
    return source_name ,module 


def _build_verify_prompt_context (context_pool :list [dict ])->list [dict ]:
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


def _normalize_metrics (raw :dict [str ,Any ])->dict [str ,Any ]:
    citation =raw .get ("citation",{})or {}
    hallucination =raw .get ("hallucination",{})or {}
    logic =raw .get ("logic",{})or {}
    completeness =raw .get ("completeness",{})or {}
    fmt =raw .get ("format",{})or {}

    return {
    "citation":{
    "score":_clamp_score_0_5 (citation .get ("score"),default =0.0 ),
    "missing":_as_bool (citation .get ("missing"),default =True ),
    "fabricated":_as_bool (citation .get ("fabricated"),default =False ),
    },
    "hallucination":{
    "score":_clamp_score_0_5 (hallucination .get ("score"),default =0.0 ),
    "unsupported_claim":_as_bool (
    hallucination .get ("unsupported_claim"),
    default =False ,
    ),
    },
    "logic":{
    "score":_clamp_score_0_5 (logic .get ("score"),default =0.0 ),
    "contradiction":_as_bool (logic .get ("contradiction"),default =False ),
    },
    "completeness":{
    "score":_clamp_score_0_5 (completeness .get ("score"),default =0.0 ),
    },
    "format":{
    "score":_clamp_score_0_5 (fmt .get ("score"),default =0.0 ),
    },
    "confidence":_clamp_confidence (raw .get ("confidence"),default =0.0 ),
    }


def _weighted_total_score (metrics :dict [str ,Any ])->float :
    w_citation =float (AppConfig .SV_WEIGHT_CITATION )
    w_hallucination =float (AppConfig .SV_WEIGHT_HALLUCINATION )
    w_logic =float (AppConfig .SV_WEIGHT_LOGIC )
    w_completeness =float (AppConfig .SV_WEIGHT_COMPLETENESS )
    w_format =float (AppConfig .SV_WEIGHT_FORMAT )
    w_sum =w_citation +w_hallucination +w_logic +w_completeness +w_format 
    if w_sum <=0 :
        w_citation ,w_hallucination ,w_logic ,w_completeness ,w_format =(
        0.35 ,
        0.25 ,
        0.20 ,
        0.15 ,
        0.05 ,
        )
        w_sum =1.0 
    else :
        w_citation /=w_sum 
        w_hallucination /=w_sum 
        w_logic /=w_sum 
        w_completeness /=w_sum 
        w_format /=w_sum 

    total =(
    metrics ["citation"]["score"]*w_citation 
    +metrics ["hallucination"]["score"]*w_hallucination 
    +metrics ["logic"]["score"]*w_logic 
    +metrics ["completeness"]["score"]*w_completeness 
    +metrics ["format"]["score"]*w_format 
    )
    return max (0.0 ,min (5.0 ,float (total )))


def _normalize_failure_type (value :object )->str :
    text =str (value or "").strip ().upper ()
    if text not in ALLOWED_FAILURE_TYPES :
        return "FORMAT_ERROR"
    return text 


def _extract_missing_entities (response_text :str )->set [str ]:
    text =str (response_text or "")
    m =_MISSING_EVIDENCE_PATTERN .search (text )
    if not m :
        return set ()
    raw =m .group (1 )
    items :set [str ]=set ()
    for part in raw .split (","):
        token =str (part or "").strip ().strip (".")
        if token :
            items .add (token )
    return items 


def _companies_from_text (text :str )->set [str ]:
    lowered =str (text or "").lower ()
    found :set [str ]=set ()
    for alias ,company in _COMPANY_ALIAS_MAP .items ():
        if alias in lowered :
            found .add (company )
    return found 


def _missing_citation_ids (
citations :list [dict ],
context_pool :list [dict ],
)->list [str ]:
    context_ids ={str (doc .get ("id","")).strip ()for doc in context_pool if str (doc .get ("id","")).strip ()}
    missing :set [str ]=set ()
    for citation in citations :
        citation_id =str ((citation or {}).get ("id","")).strip ()
        if not citation_id or citation_id not in context_ids :
            missing .add (citation_id or "<empty-id>")
    return sorted (missing )


def _decide_strict_action (
metrics :dict [str ,Any ],
total_score :float ,
)->tuple [str ,str ,str ]:
# Level One: Fatal Boolean Interception
    if bool (metrics ["citation"]["missing"])and bool (AppConfig .SV_BLOCK_CITATION_MISSING ):
        trigger ="citation_missing"
        return ("REPAIR",trigger ,_normalize_failure_type (_TRIGGER_TO_FAILURE_TYPE [trigger ]))
    if bool (metrics ["citation"]["fabricated"]):
        trigger ="citation_fabricated"
        return ("REPAIR",trigger ,_normalize_failure_type (_TRIGGER_TO_FAILURE_TYPE [trigger ]))
    if bool (metrics ["hallucination"]["unsupported_claim"])and bool (
    AppConfig .SV_BLOCK_UNSUPPORTED_CLAIM 
    ):
        trigger ="unsupported_claim"
        return ("REPAIR",trigger ,_normalize_failure_type (_TRIGGER_TO_FAILURE_TYPE [trigger ]))
    if bool (metrics ["logic"]["contradiction"]):
        trigger ="logic_contradiction"
        return ("REPAIR",trigger ,_normalize_failure_type (_TRIGGER_TO_FAILURE_TYPE [trigger ]))

        # Second level: lowest score interception in one dimension
    floor =float (AppConfig .SV_CRITICAL_SCORE_FLOOR )
    if float (metrics ["citation"]["score"])<=floor :
        trigger ="citation_score_too_low"
        return ("REPAIR",trigger ,_normalize_failure_type (_TRIGGER_TO_FAILURE_TYPE [trigger ]))
    if float (metrics ["hallucination"]["score"])<=floor :
        trigger ="hallucination_score_too_low"
        return ("REPAIR",trigger ,_normalize_failure_type (_TRIGGER_TO_FAILURE_TYPE [trigger ]))
    if float (metrics ["logic"]["score"])<=floor :
        trigger ="logic_score_too_low"
        return ("REPAIR",trigger ,_normalize_failure_type (_TRIGGER_TO_FAILURE_TYPE [trigger ]))

        # Level 3: Weighted Total Score
    total_threshold =float (AppConfig .SV_TOTAL_THRESHOLD )
    if total_score <total_threshold :
        trigger ="total_score_below_threshold"
        return ("REPAIR",trigger ,_normalize_failure_type (_TRIGGER_TO_FAILURE_TYPE [trigger ]))

    return ("PASS","","")


def strict_verify_node (state :AgentState )->AgentState :
    full_context_pool =state .get ("context_pool",[])or []
    verify_context_pool =_pick_verify_context (full_context_pool )
    verify_prompt_context_pool =_build_verify_prompt_context (verify_context_pool )
    prompt_state :AgentState =dict (state )
    prompt_state ["context_pool"]=verify_prompt_context_pool 
    out =run_prompt (StrictVerifyPrompt ,prompt_state )

    metrics =_normalize_metrics (out )
    deterministic_notes :list [str ]=[]

    # Optional relaxation: if system already has non-empty citations, do not treat
    # model-side "citation.missing=true" as a hard blocker.
    if bool (metrics ["citation"]["missing"])and bool (
    AppConfig .SV_ALLOW_NONEMPTY_CITATIONS_OVERRIDE_MISSING 
    ):
        if len (state .get ("citations",[])or [])>0 :
            metrics ["citation"]["missing"]=False 
            metrics ["citation"]["score"]=max (float (metrics ["citation"]["score"]),2.0 )
            deterministic_notes .append ("override_citation_missing_by_nonempty_citations")

    alignment =_numeric_alignment_summary (
    query =str (state .get ("query","")),
    resolved_query =str (state .get ("resolved_query","")),
    response =str (state .get ("response","")),
    evidence_table =state .get ("evidence_table",[])or [],
    )
    if bool (metrics ["logic"]["contradiction"])and bool (alignment .get ("strong_alignment")):
    # Deterministic numeric alignment proves no cross-company/year contradiction.
        metrics ["logic"]["contradiction"]=False 
        metrics ["logic"]["score"]=max (float (metrics ["logic"]["score"]),3.0 )
        deterministic_notes .append ("override_logic_contradiction_by_numeric_alignment")

    total_score =_weighted_total_score (metrics )
    missing_entities =_extract_missing_entities (str (state .get ("response","")))
    if missing_entities :
        strict_action ="REPAIR"
        repair_trigger ="insufficient_evidence_missing_company"
        failure_type ="INSUFFICIENT_EVIDENCE"
    else :
        strict_action ,repair_trigger ,failure_type =_decide_strict_action (
        metrics ,
        total_score ,
        )

    missing_citation_ids =_missing_citation_ids (
    citations =state .get ("citations",[])or [],
    context_pool =full_context_pool ,
    )
    if missing_citation_ids :
        deterministic_notes .append (
        f"missing_citation_ids={', '.join(missing_citation_ids[:6])}"
        )
        if bool (AppConfig .SV_BLOCK_CITATION_NOT_IN_CONTEXT ):
            strict_action ="REPAIR"
            repair_trigger ="citation_not_in_context"
            failure_type =_normalize_failure_type (_TRIGGER_TO_FAILURE_TYPE [repair_trigger ])

    numeric_mismatches =list (alignment .get ("mismatches",[])or [])
    if numeric_mismatches :
        strict_action ="REPAIR"
        repair_trigger ="numeric_year_value_mismatch"
        failure_type =_normalize_failure_type (_TRIGGER_TO_FAILURE_TYPE [repair_trigger ])
        deterministic_notes .append (
        "numeric_year_value_mismatch="+" | ".join (numeric_mismatches [:3 ])
        )

    numeric_citation_mismatches =_detect_numeric_claim_citation_mismatch (
    query =str (state .get ("query","")),
    resolved_query =str (state .get ("resolved_query","")),
    response =str (state .get ("response","")),
    citations =state .get ("citations",[])or [],
    context_pool =full_context_pool ,
    )
    if numeric_citation_mismatches and repair_trigger !="numeric_year_value_mismatch":
        min_mismatch_count =max (1 ,int (AppConfig .SV_NUMERIC_CITATION_MISMATCH_MIN_COUNT ))
        if len (numeric_citation_mismatches )>=min_mismatch_count :
            strict_action ="REPAIR"
            repair_trigger ="numeric_claim_not_in_citation"
            failure_type =_normalize_failure_type (_TRIGGER_TO_FAILURE_TYPE [repair_trigger ])
            deterministic_notes .append (
            "numeric_claim_not_in_citation="+" | ".join (numeric_citation_mismatches [:3 ])
            )
        else :
            deterministic_notes .append (
            f"numeric_claim_not_in_citation_tolerated(count={len(numeric_citation_mismatches)}<{min_mismatch_count})"
            )
    missing_numeric_companies =sorted (alignment .get ("missing_in_response",set ())or set ())
    if missing_numeric_companies :
        deterministic_notes .append (
        "numeric_expected_but_missing_in_response="+", ".join (missing_numeric_companies [:6 ])
        )

    if failure_type =="CITATION_MISMATCH"and missing_entities :
        failure_type ="INSUFFICIENT_EVIDENCE"
        repair_trigger ="insufficient_evidence_missing_company"

    verdict ="PASS"if strict_action =="PASS"else "FAIL"
    strict_score =max (0.0 ,min (1.0 ,total_score /5.0 ))
    reason =(
    f"trigger={repair_trigger or 'none'}; total_score={total_score:.2f}; "
    f"citation={metrics['citation']['score']:.2f}, "
    f"hallucination={metrics['hallucination']['score']:.2f}, "
    f"logic={metrics['logic']['score']:.2f}, "
    f"completeness={metrics['completeness']['score']:.2f}, "
    f"format={metrics['format']['score']:.2f}"
    )
    if missing_entities :
        reason +=f"; missing_entities={', '.join(sorted(missing_entities))}"
    if deterministic_notes :
        reason +=f"; deterministic={' | '.join(deterministic_notes)}"

    state ["strict_verdict"]=verdict 
    state ["strict_score"]=strict_score 
    state ["strict_total_score"]=total_score 
    state ["strict_action"]=strict_action 
    state ["strict_metrics"]=metrics 
    state ["strict_confidence"]=float (metrics .get ("confidence",0.0 ))
    state ["repair_trigger"]=repair_trigger 
    state ["strict_reason"]=reason 
    state ["verified_revision"]=state .get ("response_revision",0 )
    if verdict =="FAIL":
        state ["failure_type"]=failure_type 
    else :
        state .pop ("failure_type",None )
        state ["repair_trigger"]=""

    state .setdefault ("steps_log",[]).append (
    StepLog (
    node ="strict_verify",
    info ={
    "state":{
    "response_preview":clip_text (state .get ("response",""),180 ),
    "context_pool_count":len (full_context_pool ),
    "verify_context_count":len (verify_context_pool ),
    "verify_context_chars":_context_chars (verify_context_pool ),
    "verify_prompt_context_chars":_context_chars (verify_prompt_context_pool ),
    "response_revision":state .get ("response_revision",0 ),
    "verified_revision":state .get ("verified_revision",0 ),
    },
    "llm_input":{
    "context_pool_preview":preview_docs (verify_prompt_context_pool ),
    },
    "llm_output":{
    "metrics":metrics ,
    "verdict":verdict ,
    "action":strict_action ,
    "total_score":total_score ,
    "score":strict_score ,
    "score_threshold":float (AppConfig .SV_TOTAL_THRESHOLD ),
    "repair_trigger":repair_trigger ,
    "failure_type":failure_type if verdict =="FAIL"else "",
    "deterministic_notes":deterministic_notes ,
    "reason_preview":clip_text (reason ,220 ),
    },
    },
    timestamp =time .time (),
    )
    )
    return state 
