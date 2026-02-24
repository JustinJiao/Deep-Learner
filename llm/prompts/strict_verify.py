from llm.prompts.base import PromptContract


class StrictVerifyPrompt(PromptContract):
    READS = ["query", "response", "citations", "context_pool"]
    WRITES = ["score", "verdict", "failure_type", "reason"]

    SYSTEM = """
你是 Deep-Learner 的严格验证器。

请判断 response 是否可被 context_pool 充分支撑。

输出字段：
- score: 0~1（核心判定字段，分数越高表示证据支撑越充分）
- verdict: PASS | FAIL（兼容字段）
- failure_type:
  - INSUFFICIENT_EVIDENCE
  - LOGICAL_ERROR
  - CITATION_MISMATCH
  - FORMAT_ERROR
- reason: 失败或通过原因

规则：
- score 必须与 verdict 一致：通常 score>=0.75 时 verdict=PASS，否则 verdict=FAIL。
- 当 verdict=PASS 时，failure_type 可填任意占位值。
- 当 verdict=FAIL 时，failure_type 必须准确分类。
- 若 response 为“不确定/uncertain”，但 context_pool 已包含可直接回答的证据，应判为 LOGICAL_ERROR，而非 INSUFFICIENT_EVIDENCE。

必须只输出 JSON：
{
  "score": 0.0,
  "verdict": "PASS" | "FAIL",
  "failure_type": "INSUFFICIENT_EVIDENCE" | "LOGICAL_ERROR" | "CITATION_MISMATCH" | "FORMAT_ERROR",
  "reason": "..."
}
    """

    def build_user_prompt(self, state):
        docs = state.get("context_pool", []) or []
        formatted_docs = ""
        for i, d in enumerate(docs):
            formatted_docs += f"""
文档 {i + 1}
ID: {d.get('id', '')}
Title: {d.get('title', '')}
Score: {d.get('score', 0.0)}
内容:
{d.get('content', '')}
--------------------
"""

        citations = state.get("citations", []) or []
        formatted_citations = ""
        for i, c in enumerate(citations):
            formatted_citations += f"""
引用 {i + 1}
ID: {c.get('id', '')}
Title: {c.get('title', '')}
Score: {c.get('score', 0.0)}
Quote:
{c.get('quote', '')}
--------------------
"""

        return f"""
用户问题:
{state.get("query", "")}

回答:
{state.get("response", "")}

引用:
{formatted_citations}

检索文档:
{formatted_docs}
"""
