import pytest

from core.errors import InvalidStateTransitionError
from core.executor import _assert_forward_only


def test_forward_transition_allowed():
    _assert_forward_only("MEMORY", "PHASE1")
    _assert_forward_only("PHASE1", "FINALIZE")


def test_backward_transition_forbidden():
    with pytest.raises(InvalidStateTransitionError):
        _assert_forward_only("PHASE2", "PHASE1")


def test_non_whitelisted_transition_forbidden():
    with pytest.raises(InvalidStateTransitionError):
        _assert_forward_only("MEMORY", "REPAIR")
