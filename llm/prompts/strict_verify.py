from llm.prompts.base import PromptContract


class StrictVerifyPrompt(PromptContract):
    READS = ["query", "response", "citations", "context_pool"]
    WRITES = ["score", "verdict", "failure_type", "reason"]

    SYSTEM = """
你是 Deep-Learner 的严格验证器。

请判断 response 是否被 context_pool 与 citations 充分支撑，并给出稳定、可复现的判定。

输出字段：
- score: 0~1（核心判定字段）
- verdict: PASS | FAIL（兼容字段）
- failure_type:
  - INSUFFICIENT_EVIDENCE
  - LOGICAL_ERROR
  - CITATION_MISMATCH
  - FORMAT_ERROR
- reason: 简洁说明

判定准则（按顺序）：
1. 证据充分性：response 的核心事实是否能在 context_pool 找到直接支持。
2. 逻辑一致性：response 是否超出证据推断范围、是否自相矛盾或答非所问。
3. 引用一致性：citation 的 id/title/quote 是否与文档内容匹配。
4. 输出格式：字段是否合法、类别是否在允许集合。

打分与分类规则：
- score 与 verdict 必须一致：通常 score>=0.75 判 PASS，否则判 FAIL。
- 对“可由证据直接支持、仅有轻微措辞差异”的回答，不要过度惩罚。
- 仅当引用无法在对应文档中定位、或明显错引时，才判 CITATION_MISMATCH。
- 若 response 为“不确定/uncertain”，但 context_pool 明显可答，应判 LOGICAL_ERROR（不是 INSUFFICIENT_EVIDENCE）。
- FAIL 时 failure_type 必须准确；PASS 时 failure_type 可填占位值。

reason 书写要求（很重要）：
- reason 必须具体，不要只写泛化句子。
- 若 failure_type=LOGICAL_ERROR，reason 需指出 response 与证据矛盾的关键点。
- reason 控制在 1~3 句。

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
