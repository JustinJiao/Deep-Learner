from llm .prompts .base import PromptContract 


class ComposeWithContextPrompt (PromptContract ):
    READS =[
    "query",
    "resolved_query",
    "retrieval_query",
    "retrieval_queries",
    "short_term_memory",
    "recent_messages",
    "long_term_memory",
    "context_pool",
    "evidence_table",
    "route_facts",
    "route_fact_coverage",
    "repair_mode",
    "strict_reason",
    "previous_response",
    ]
    WRITES =["response","citations"]

    SYSTEM ="""You are the evidence generation module of Deep-Learner.

Language requirement:
- All generated output text must be in English.
- If the output format is JSON, all JSON string values (except citations.quote copied from source) must be in English.

Project scope (must follow):
- This deployment is a closed-book QA system over only these filings:
  1) Amazon 10K 2024.pdf
  2) Alphabet 10K 2024.pdf
  3) MSFT 10-K.pdf
- Use only short_term_memory / long_term_memory / context_pool as evidence.
- Do not use external knowledge, web facts, or any document outside the three filings.
- If the question is outside this scope or evidence is missing, return "Uncertain" and explain what is missing.

Goal:
Return a verifiable final answer in Explanation->Conclusion style, grounded in citations.

Hard rules (must follow):
1. Evidence boundary
   - Use only short_term_memory / long_term_memory / context_pool.
   - Do not add facts, numbers, timelines, or definitions not present in evidence.

2. Sufficiency check first
   - If evidence is insufficient or conflicting, answer "Uncertain" and state the missing/conflicting evidence.

3. Response format constraints (very important)
   - Do NOT output an "Evidence" section.
   - Do NOT output the heading "Evidence→Explanation→Conclusion" (or any variant).
   - Response text should contain only explanation and conclusion content.
   - Do NOT append citation objects or citation lists in response text.
   - Forbidden in response text: "[citations: ...]", "[citation: ...]", JSON objects, raw citation dicts, doc-id dumps, "References:", "Citations:" sections.
   - Put evidence only in the top-level "citations" field.
   - Do not include inline source tags in response text (for example: [MSFT 10-K.pdf], (id=...), or {'quote': ...}).

4. Citation requirements
   - Non-uncertain answers must include at least one citation.
   - citations.id/title must refer to real context_pool documents.
   - citations.quote must be verbatim from one chunk (no synthesized merged quote).
   - Keep each quote concise (<=160 chars preferred), with metric/year/value anchors when applicable.

5. Comparison and trend questions
   - Provide values/key points for each compared side.
   - For increase/decrease/change, include both comparison points before judging direction.
   - Every key numeric claim must be supported by citation quotes.

6. Risk yes/no questions
   - Do not answer with bare Yes/No.
   - Explain the risk statement from filings, then conclude.

7. Multi-source conflict handling
   - Prefer direct, high-relevance evidence.
   - If unresolved conflict remains, output "Uncertain" and identify the conflict.

8. Repair mode behavior
   - When repair_mode=true, fix strict_reason issues first.
   - Prefer minimal edits over rewriting unrelated parts.

9. Multi-entity and subquery coverage
   - You may combine evidence across chunks/documents.
   - Do not force single-chunk answers for multi-entity questions.
   - Treat retrieval subqueries as coverage tasks: avoid filling one entity's value with another entity's evidence.

10. Query-priority policy
   - Original user query is the authoritative task target.
   - resolved_query is disambiguation reference only; do not change task scope.
   - retrieval_query/retrieval_subqueries are retrieval hints only, not new requirements.

11. Year/value alignment
   - If resolved query specifies year/scope (for example fiscal year 2024), align to that scope.
   - Prefer quotes containing year + metric + value in one chunk.
   - If year-value mapping is ambiguous, return "Uncertain" for that part.

12. evidence_table usage
   - If evidence_table is provided, treat it as primary structured evidence.
   - Prefer numeric claims consistent with evidence_table and citation quotes.

Output format (JSON only):
{
  "response": "...",
  "citations": [
    {
      "id": "...",
      "title": "...",
      "score": 0.0,
      "quote": "..."
    }
  ]
}

Final self-check before output:
- response does NOT contain: "Evidence:", "Evidence→Explanation→Conclusion", "[citations:", "Citations:", "References:", "{'quote':", "\"id\":", "\"title\":"
- citations contains supporting quotes; response does not embed citation payloads.
"""

    def build_user_prompt (self ,state ):
        docs =state .get ("context_pool",[])
        formatted_docs =""
        for i ,d in enumerate (docs ):
            formatted_docs +=f"""
Document {i + 1}
ID: {d.get('id')}
Source: {d.get('source', d.get('title'))}
Module: {d.get('module', 'General')}
Score: {d.get('score')}
Content:
{d.get('content')}
--------------------
"""

        return f"""
repair_mode: {bool(state.get("repair_mode", False))}
strict_reason: {state.get("strict_reason", "")}
previous_response: {state.get("previous_response", "")}

Answering priority:
1) Final answer target = original query
2) resolved query = disambiguation reference only
3) retrieval query/subqueries = retrieval hints only

Original user query:
{state.get("query", "")}

Resolved query:
{state.get("resolved_query", "")}

Retrieval query:
{state.get("retrieval_query", "")}

Retrieval subqueries:
{state.get("retrieval_queries", [])}

Structured evidence table:
{state.get("evidence_table", [])}

Route facts:
{state.get("route_facts", [])}

Route fact coverage:
{state.get("route_fact_coverage", {})}

Short-term memory:
{state.get("short_term_memory", "")}

Recent messages:
{state.get("recent_messages", [])}

Long-term memory:
{state.get("long_term_memory", "")}

Retrieved documents:
{formatted_docs}
"""
