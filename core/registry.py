from nodes.intent import intent_node
from nodes.planner import planner_node
from nodes.recall_ltm import recall_ltm_node
from nodes.query_rewrite import query_rewrite_node
from nodes.retrieve import retrieve_node
from nodes.compose import compose_node
from nodes.verify import verify_node
from nodes.repair import repair_node
from nodes.persist_ltm import persist_ltm_node
from nodes.finalize import finalize_node
from nodes.stm_read import stm_read_node
from nodes.stm_write import stm_write_node
from nodes.stm_summary import stm_summary_node

NODE_REGISTRY = {
    "stm_read": stm_read_node,
    "intent": intent_node,
    "planner": planner_node,

    "recall_ltm": recall_ltm_node,
    "query_rewrite": query_rewrite_node,
    "retrieve": retrieve_node,
    "compose": compose_node,
    "verify": verify_node,
    "repair": repair_node,

    "finalize": finalize_node,
    "stm_write": stm_write_node,
    "stm_summary": stm_summary_node,
    "persist_ltm": persist_ltm_node,
}
