# core/executor.py
from __future__ import annotations

from typing import Any

from config.settings import AppConfig
from core.registry import NODE_REGISTRY
from core.errors import UnknownNodeError
from core.state import AgentState, StepLog


def _append_log(state: AgentState, log: StepLog) -> None:
    logs = state.setdefault("steps_log", [])
    logs.append(log)
    cap = int(getattr(AppConfig, "MAX_STEPS_LOG", 200))
    if cap > 0 and len(logs) > cap:
        # 保留尾部，避免 state 膨胀
        state["steps_log"] = logs[-cap:]


class AgentExecutor:
    """生产级单轮执行入口（具备降级与防污染写入）。"""

    def run(self, session_id: str, query: str) -> AgentState:
        state: AgentState = {
            "session_id": session_id,
            "query": query,
            "steps_log": [],
            "loop_count": 0,
            "run_status": "running",  # running | ok | degraded | error
        }

        try:
            # 前置固定节点：全部走 registry，避免命名不一致
            state = NODE_REGISTRY["stm_read"](state)
            state = NODE_REGISTRY["intent"](state)
            state = NODE_REGISTRY["planner"](state)

            plan = state["plan"]

            while not plan.is_finished():
                step = plan.current_step()
                node_fn = NODE_REGISTRY.get(step)
                if node_fn is None:
                    raise UnknownNodeError(step)

                state = node_fn(state)

                if step == "verify" and state.get("is_hallucination", False):
                    state["loop_count"] += 1
                    _append_log(state, StepLog(node="executor", info=f"verify_fail loop={state['loop_count']}"))

                    # 超过上限：生产级降级，而不是抛异常
                    if state["loop_count"] > plan.max_loops:
                        state["run_status"] = "degraded"
                        state["response"] = (
                            "我刚才的回答缺少可靠依据，无法在当前资料范围内自洽验证。\n"
                            "为了避免误导，请你补充：你希望我基于哪些具体文档/数据，或提供更多上下文。\n"
                            "（我也可以先给一个不确定但可能的方向，并明确标注假设。）"
                        )
                        state["is_hallucination"] = False
                        _append_log(state, StepLog(node="executor", info="max_loops reached -> degraded response"))
                        plan.finish()
                        break

                    state = NODE_REGISTRY["repair"](state)
                    continue

                plan.advance()

            if state.get("run_status") == "running":
                state["run_status"] = "ok"

        except Exception as e:
            # 捕获所有异常：线上不崩溃，返回可诊断信息
            state["run_status"] = "error"
            state["error"] = {"type": type(e).__name__, "message": str(e)}
            _append_log(state, StepLog(node="executor", info=f"exception: {type(e).__name__}: {e}"))

            # 如果没有 response，则给一个安全兜底（不写入 memory）
            if not state.get("response"):
                state["response"] = "系统在处理时发生错误，我无法完成本次回答。请重试或提供更多信息。"

        finally:
            # 只在“正常/降级”且存在 response 时写入 STM/LTM，避免把错误尝试污染记忆
            should_persist = bool(state.get("response")) and state.get("run_status") in ("ok", "degraded")
            for tail in ("finalize", "stm_write", "stm_summary", "persist_ltm"):
                if tail in ("stm_write", "stm_summary", "persist_ltm") and not should_persist:
                    continue
                try:
                    state = NODE_REGISTRY[tail](state)
                except Exception as e:
                    _append_log(state, StepLog(node=tail, info=f"error: {type(e).__name__}: {e}"))

        return state
