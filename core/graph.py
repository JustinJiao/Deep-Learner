# core/graph.py

ALLOWED_TRANSITIONS ={
"START":{"intent"},
"intent":{"planner"},

# After the planner generates the plan, the executor enters the steps loop
"planner":{"memory_read","query_rewrite","retrieve","compose","verify","repair","finalize"},

"memory_read":{"query_rewrite"},
"query_rewrite":{"retrieve"},
"retrieve":{"compose"},
"compose":{"verify","finalize"},

"verify":{"repair","finalize"},
"repair":{"query_rewrite","retrieve","compose","finalize"},

# Closing (your executor will be called at the end)
"finalize":{"memory_write"},
"memory_write":set (),
}
