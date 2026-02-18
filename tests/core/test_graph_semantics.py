from core.graph import ALLOWED_TRANSITIONS


def test_verify_transitions():
    assert "repair" in ALLOWED_TRANSITIONS["verify"]
    assert "finalize" in ALLOWED_TRANSITIONS["verify"]


def test_no_backwards_jump():
    assert "intent" not in ALLOWED_TRANSITIONS["compose"]
