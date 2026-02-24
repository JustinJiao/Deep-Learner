from llm.prompts.base import PromptContract


class VerifyMemoryPrompt(PromptContract):
    READS = ["query", "draft_answer", "draft_confidence", "used_memory_chunks"]
    WRITES = ["score", "verdict", "reason", "risk_level"]

    SYSTEM = """
你是 Deep-Learner 的记忆覆盖验证器。

请判断仅基于记忆生成的 draft_answer 是否足以直接回答用户问题。

判定规则：
- SUFFICIENT：记忆证据足够，可直接给出答案
- NEED_RETRIEVE：记忆不足，必须进入外部检索
- score 为 0~1，分数越高表示“仅凭记忆即可回答”的可信度越高
- 建议 score>=0.70 时判为 SUFFICIENT，否则判为 NEED_RETRIEVE

risk_level 只能是：
- LOW
- MEDIUM
- HIGH

必须只输出 JSON：
{
  "score": 0.0,
  "verdict": "SUFFICIENT" | "NEED_RETRIEVE",
  "reason": "...",
  "risk_level": "LOW" | "MEDIUM" | "HIGH"
}
"""

    def build_user_prompt(self, state):
        return f"""
用户问题:
{state.get("query", "")}

记忆草答:
{state.get("draft_answer", "")}

草答置信度:
{state.get("draft_confidence", 0.0)}

使用记忆片段数:
{state.get("used_memory_chunks", 0)}
"""
