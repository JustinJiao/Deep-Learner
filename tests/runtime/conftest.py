import pytest

from config.settings import AppConfig


@pytest.fixture(autouse=True)
def _default_legacy_runtime(monkeypatch):
    # Runtime V2 is globally enabled by default; keep legacy runtime tests stable
    # unless a test explicitly opts into V2.
    monkeypatch.setattr(AppConfig, "RUNTIME_V2_ENABLED", False)
