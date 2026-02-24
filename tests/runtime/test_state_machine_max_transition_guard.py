import pytest

import core.executor as executor_mod
from config.settings import AppConfig
from core.errors import RuntimeMaxTransitionExceededError
from core.state import build_initial_state


def test_max_transition_guard(monkeypatch):
    monkeypatch.setattr(AppConfig, "RUNTIME_MAX_TRANSITIONS", 1)
    state = build_initial_state(session_id="s1", query="q1")

    executor_mod._transition(state, "PHASE1", reason="first")
    with pytest.raises(RuntimeMaxTransitionExceededError):
        executor_mod._transition(state, "FINALIZE", reason="second")
