# core/graph.py

ALLOWED_TRANSITIONS = {
    "START": {"intent"},
    "intent": {"planner"},

    # planner 生成 plan 后，executor 进入 steps loop
    "planner": {"memory_read", "query_rewrite", "retrieve", "compose", "verify", "repair", "finalize"},

    "memory_read": {"query_rewrite"},
    "query_rewrite": {"retrieve"},
    "retrieve": {"compose"},
    "compose": {"verify", "finalize"},

    "verify": {"repair", "finalize"},
    "repair": {"query_rewrite", "retrieve", "compose", "finalize"},

    # 收尾（你 executor 会在最后固定调用）
    "finalize": {"memory_write"},
    "memory_write": set(),
}
