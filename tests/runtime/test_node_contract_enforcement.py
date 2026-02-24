import pytest

import core.executor as executor_mod
from config.settings import AppConfig
from core.errors import NodeContractViolationError
from core.registry import NodeContract


def test_execute_node_blocks_unexpected_writes_when_strict(monkeypatch):
    def bad_node(state):
        state["allowed"] = 1
        state["not_allowed"] = 2
        return state

    monkeypatch.setattr(executor_mod, "NODE_REGISTRY", {"bad": bad_node})
    monkeypatch.setattr(
        executor_mod,
        "NODE_CONTRACTS",
        {
            "bad": NodeContract(
                name="bad",
                reads={"query"},
                writes={"allowed"},
                strict=True,
            )
        },
    )
    monkeypatch.setattr(AppConfig, "RUNTIME_ENFORCE_CONTRACT", True)

    with pytest.raises(NodeContractViolationError):
        executor_mod.AgentExecutor()._execute_node({"query": "x"}, "bad")
