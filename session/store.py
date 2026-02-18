# session/store.py
from session.context import SessionContext

_SESSION_STORE: dict[str, SessionContext] = {}


def get_session(session_id: str) -> SessionContext:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = SessionContext(session_id)
    return _SESSION_STORE[session_id]


def save_session(session_id: str, ctx: SessionContext):
    _SESSION_STORE[session_id] = ctx
