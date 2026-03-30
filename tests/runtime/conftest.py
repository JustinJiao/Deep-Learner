import pytest

from config.settings import AppConfig


@pytest.fixture(autouse=True)
def _default_v2_runtime(monkeypatch):
    # Keep tests aligned with the current production flow (Runtime V2).
    monkeypatch.setattr(AppConfig, "RUNTIME_V2_ENABLED", True)
