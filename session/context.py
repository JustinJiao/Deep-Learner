# session/context.py

from core.state import STMState


class SessionContext:

    def __init__(self, session_id: str):
        self.session_id = session_id

        # 初始化 STMState（匹配新版结构）
        self.stm: STMState = {
            "summary": [],
            "messages": [],
            "recent_messages": [],
            "compressed_until": 0,
        }
