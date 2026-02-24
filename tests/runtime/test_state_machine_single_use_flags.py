import pytest

from core.errors import InvalidStateTransitionError
from core.executor import _guard_phase2_once, _guard_repair_once


def test_phase2_guard_allows_once_then_blocks():
    state = {"phase2_used": False}
    _guard_phase2_once(state)

    state["phase2_used"] = True
    with pytest.raises(InvalidStateTransitionError):
        _guard_phase2_once(state)


def test_repair_guard_allows_once_then_blocks():
    state = {"repair_used": False}
    _guard_repair_once(state)

    state["repair_used"] = True
    with pytest.raises(InvalidStateTransitionError):
        _guard_repair_once(state)
