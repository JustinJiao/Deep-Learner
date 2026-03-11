from nodes.compose_with_context import _ensure_compose_route_coverage
from nodes.rewrite_query_for_retrieval import rewrite_query_for_retrieval_node


def test_rewrite_query_uses_prompt_generated_subqueries(monkeypatch):
    def fake_run_prompt(_prompt_cls, _state):
        return {
            "retrieval_queries": [
                "Microsoft cloud revenue fiscal year 2024",
                "Amazon AWS revenue fiscal year 2024",
                "Alphabet Google Cloud revenue fiscal year 2024",
            ]
        }

    monkeypatch.setattr("nodes.rewrite_query_for_retrieval.run_prompt", fake_run_prompt)

    state = {
        "query": "Compare cloud revenue of Microsoft, Amazon and Alphabet in 2024.",
        "resolved_query": "Compare cloud revenue of Microsoft, Amazon and Alphabet in fiscal year 2024.",
        "long_term_memory": "",
        "recent_messages": [],
        "steps_log": [],
    }

    out = rewrite_query_for_retrieval_node(state)
    queries = out.get("retrieval_queries", [])

    assert queries == [
        "Microsoft cloud revenue fiscal year 2024",
        "Amazon AWS revenue fiscal year 2024",
        "Alphabet Google Cloud revenue fiscal year 2024",
    ]
    assert out.get("retrieval_query") == queries[0]


def test_rewrite_query_legacy_single_query_fallback(monkeypatch):
    def fake_run_prompt(_prompt_cls, _state):
        return {
            "retrieval_query": (
                "Compare cloud revenue segments of Amazon, Microsoft, and Alphabet "
                "in fiscal year 2024"
            )
        }

    monkeypatch.setattr("nodes.rewrite_query_for_retrieval.run_prompt", fake_run_prompt)
    monkeypatch.setattr("nodes.rewrite_query_for_retrieval.AppConfig.RETRIEVAL_MULTI_QUERY_MAX_QUERIES", 4)

    state = {
        "query": (
            "Compare the cloud-related revenue segments of Microsoft, Amazon, and Alphabet. "
            "Which cloud business generated the highest revenue in the most recent fiscal year?"
        ),
        "resolved_query": (
            "Compare the cloud-related revenue segments of Microsoft, Amazon, and Alphabet "
            "for fiscal year 2024."
        ),
        "long_term_memory": "project background: latest fiscal year is 2024",
        "recent_messages": [],
        "steps_log": [],
    }

    out = rewrite_query_for_retrieval_node(state)
    queries = out.get("retrieval_queries", [])

    assert queries == [
        "Compare cloud revenue segments of Amazon, Microsoft, and Alphabet in fiscal year 2024"
    ]


def test_compose_route_coverage_injects_missing_route_docs():
    compose_context_pool = [
        {"id": "msft-1", "content": "microsoft doc", "metadata": {}, "score": 1.0},
    ]
    full_context_pool = [
        {"id": "msft-1", "content": "microsoft doc", "metadata": {}, "score": 1.0},
        {"id": "amzn-1", "content": "amazon doc", "metadata": {}, "score": 0.9},
        {"id": "goog-1", "content": "alphabet doc", "metadata": {}, "score": 0.8},
    ]
    phase1_query_routes = [
        {"route_index": 0, "query": "microsoft", "candidates": [{"id": "msft-1"}]},
        {"route_index": 1, "query": "amazon", "candidates": [{"id": "amzn-1"}]},
        {"route_index": 2, "query": "alphabet", "candidates": [{"id": "goog-1"}]},
    ]

    out_pool, missing_routes = _ensure_compose_route_coverage(
        compose_context_pool=compose_context_pool,
        full_context_pool=full_context_pool,
        phase1_query_routes=phase1_query_routes,
    )

    out_ids = [doc.get("id") for doc in out_pool]
    assert "msft-1" in out_ids
    assert "amzn-1" in out_ids
    assert "goog-1" in out_ids
    assert set(missing_routes) == {1, 2}
