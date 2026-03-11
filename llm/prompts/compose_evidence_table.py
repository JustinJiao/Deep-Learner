from llm.prompts.base import PromptContract


class ComposeEvidenceTablePrompt(PromptContract):
    READS = [
        "query",
        "resolved_query",
        "retrieval_query",
        "retrieval_queries",
        "context_pool",
    ]
    WRITES = ["evidence_table"]

    SYSTEM = """
You are the evidence-table extraction module for Deep-Learner.

Language requirement:
- Output must be English-only.
- Output JSON only. No markdown. No extra keys.

Project scope:
- Use only context_pool documents.
- Do not use external knowledge.

Task:
- Extract company-level numeric evidence rows that can support the final answer.
- Focus on comparison questions with company/metric/year/value.
- If the query has an explicit year (for example fiscal year 2024), prefer rows aligned to that year.
- Keep quote verbatim from one single context chunk.

Output JSON schema:
{
  "evidence_table": [
    {
      "company": "Amazon | Microsoft | Alphabet | ...",
      "metric": "short metric name",
      "fiscal_year": 2024,
      "fiscal_period_end": "June 30, 2024 | December 31, 2024 | unknown",
      "value": "107,556",
      "unit": "million | billion | percent | raw",
      "citation_id": "context doc id",
      "citation_title": "source title",
      "quote": "verbatim snippet from that doc",
      "confidence": 0.0
    }
  ]
}

Rules:
- Return up to 12 rows, highest quality first.
- Do not invent company/year/value.
- If evidence is ambiguous, skip that row.
- citation_id must come from provided context docs.
- If the quote includes period wording such as \"Year Ended June 30, 2024\", fill fiscal_period_end with that date.
"""

    def build_user_prompt(self, state):
        docs = state.get("context_pool", []) or []
        formatted_docs = ""
        for i, d in enumerate(docs):
            formatted_docs += f"""
Document {i + 1}
ID: {d.get('id', '')}
Source: {d.get('source', d.get('title', ''))}
Module: {d.get('module', 'General')}
Score: {d.get('score', 0.0)}
Content:
{d.get('content', '')}
--------------------
"""

        return f"""
Original query:
{state.get("query", "")}

Resolved query:
{state.get("resolved_query", "")}

Retrieval query:
{state.get("retrieval_query", "")}

Retrieval subqueries:
{state.get("retrieval_queries", [])}

Context docs:
{formatted_docs}
"""
