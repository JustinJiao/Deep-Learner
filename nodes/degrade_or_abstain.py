import time 
import re 

from core .state import AgentState ,StepLog 
from nodes .log_utils import clip_text 


_FAILURE_TYPE_MESSAGES ={
"INSUFFICIENT_EVIDENCE":"Some required evidence is missing or not specific enough to support a reliable final answer.",
"LOGICAL_ERROR":"The draft answer conflicts with the retrieved evidence on at least one key point.",
"CITATION_MISMATCH":"One or more claims in the draft answer are not fully supported by the cited passages.",
"FORMAT_ERROR":"The draft answer did not satisfy required answer constraints.",
}

_TRIGGER_MESSAGES ={
"citation_missing":"Some important claims were made without adequate citation support.",
"citation_fabricated":"At least one citation could not be verified against retrieved context.",
"citation_not_in_context":"At least one cited snippet was not found in the retrieved context.",
"unsupported_claim":"The draft answer includes claims not grounded in retrieved evidence.",
"logic_contradiction":"Different parts of the draft answer conflict with each other or with evidence.",
"citation_score_too_low":"Citation quality is below the reliability threshold.",
"hallucination_score_too_low":"Evidence support is too weak for a trustworthy final answer.",
"logic_score_too_low":"Reasoning quality is below the reliability threshold.",
"total_score_below_threshold":"Overall verification score is below the acceptance threshold.",
"insufficient_evidence_missing_company":"At least one requested entity is missing direct evidence.",
"numeric_year_value_mismatch":"At least one company-year numeric value does not match citation evidence.",
"numeric_claim_not_in_citation":"At least one numeric claim is not present in the cited passages.",
}

_ROUTE_MISSING_REASON_MESSAGES ={
"no_candidates":"No relevant evidence chunk was retrieved for this sub-question.",
"numeric_value_not_found_in_best_evidence":"The best matching evidence does not contain the required numeric value.",
"best_evidence_entity_mismatch":"The top evidence appears to describe a different entity.",
"missing_entity_coverage":"No direct evidence was retrieved for this entity.",
"insufficient_route_evidence":"Evidence for this sub-question is insufficient.",
}

_MISSING_EVIDENCE_PATTERN =re .compile (
r"missing\s+explicit\s+evidence\s+for\s*:\s*([^\n\.]+)",
flags =re .IGNORECASE ,
)


def _normalize_text (value :object )->str :
    return " ".join (str (value or "").split ()).strip (" ,;:")


def _strip_inline_citation_artifacts (text :str )->str :
    raw =str (text or "")
    lowered =raw .lower ()
    markers =[
    "[citations:",
    "[citation:",
    "\ncitations:",
    "\nreferences:",
    ]
    cut_idx =-1 
    for marker in markers :
        idx =lowered .find (marker )
        if idx <0 :
            continue 
        if cut_idx <0 or idx <cut_idx :
            cut_idx =idx 
    if cut_idx >=0 :
        raw =raw [:cut_idx ]
    return raw .strip ()or "Uncertain."


def _collect_human_readable_issues (state :AgentState )->list [str ]:
    issues :list [str ]=[]

    failure_type =_normalize_text (state .get ("failure_type","")).upper ()
    if failure_type in _FAILURE_TYPE_MESSAGES :
        issues .append (_FAILURE_TYPE_MESSAGES [failure_type ])

    repair_trigger =_normalize_text (state .get ("repair_trigger","")).lower ()
    if repair_trigger in _TRIGGER_MESSAGES :
        issues .append (_TRIGGER_MESSAGES [repair_trigger ])

    route_facts =state .get ("route_facts",[])or []
    for fact in route_facts :
        if not bool (fact .get ("missing",False )):
            continue 
        entity =_normalize_text (fact .get ("entity",""))
        route_query =_normalize_text (fact .get ("route_query",""))
        missing_reason =_normalize_text (fact .get ("missing_reason","")).lower ()or "insufficient_route_evidence"
        reason_text =_ROUTE_MISSING_REASON_MESSAGES .get (
        missing_reason ,
        "Evidence for this sub-question is insufficient or conflicting.",
        )
        label =entity or route_query or f"sub-question {int (fact .get ('route_index',0 ))+1 }"
        issues .append (f"Missing evidence for {label}: {reason_text}")

    coverage =state .get ("route_fact_coverage",{})or {}
    missing_entities =list (coverage .get ("missing_entities",[])or [])
    for entity in missing_entities :
        text =_normalize_text (entity )
        if text :
            reason_text =_ROUTE_MISSING_REASON_MESSAGES ["missing_entity_coverage"]
            issues .append (f"Missing evidence for {text}: {reason_text}")

    strict_reason =str (state .get ("strict_reason","")or "")
    match =_MISSING_EVIDENCE_PATTERN .search (strict_reason )
    if match :
        raw_group =str (match .group (1 )or "")
        for token in re .split (r"[,\|;/]",raw_group ):
            item =_normalize_text (token )
            if item :
                issues .append (f"Missing direct evidence for: {item}.")

    deduped :list [str ]=[]
    seen :set [str ]=set ()
    for issue in issues :
        text =_normalize_text (issue )
        if not text :
            continue 
        key =text .lower ()
        if key in seen :
            continue 
        seen .add (key )
        deduped .append (text )
    return deduped 


def degrade_or_abstain_node (state :AgentState )->AgentState :
    reason =state .get ("strict_reason","")or state .get ("memory_reason","")
    failure_type =state .get ("failure_type","")
    repair_trigger =state .get ("repair_trigger","")
    last_compose_response =str (state .get ("response","")or state .get ("previous_response","")).strip ()
    if not last_compose_response :
        last_compose_response ="Uncertain."
    last_compose_response =_strip_inline_citation_artifacts (last_compose_response )

    issues =_collect_human_readable_issues (state )
    if not issues :
        issues =[
        "Verification found reliability risks, but no single root cause could be localized from current traces.",
        ]

    issue_lines ="\n".join (f"- {item}"for item in issues [:6 ])
    answer =(
    "❗ This answer may contain issues and should be treated as provisional.\n\n"
    "Potential problems detected:\n"
    f"{issue_lines}\n\n"
    "Based on the currently available evidence, the best conclusion we can provide is:\n"
    f"{last_compose_response}"
    )

    state ["response"]=answer 
    state ["citations"]=[]
    state ["run_status"]="degraded"
    state ["strict_status"]="FAILED"

    state .setdefault ("steps_log",[]).append (
    StepLog (
    node ="degrade_or_abstain",
    info ={
    "state":{
    "strict_reason_preview":clip_text (reason ,220 ),
    "failure_type":failure_type ,
    "repair_trigger":repair_trigger ,
    "response_preview":clip_text (answer ,220 ),
    },
    },
    timestamp =time .time (),
    )
    )
    return state 
