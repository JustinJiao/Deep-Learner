from nodes.strict_verify import strict_verify_node


def _fake_pass_metrics(_prompt_cls, _state):
    return {
        "citation": {"score": 5, "missing": False, "fabricated": False},
        "hallucination": {"score": 5, "unsupported_claim": False},
        "logic": {"score": 5, "contradiction": False},
        "completeness": {"score": 5},
        "format": {"score": 5},
        "confidence": 1.0,
    }


def _fake_contradiction_metrics(_prompt_cls, _state):
    return {
        "citation": {"score": 5, "missing": False, "fabricated": False},
        "hallucination": {"score": 0, "unsupported_claim": False},
        "logic": {"score": 0, "contradiction": True},
        "completeness": {"score": 3},
        "format": {"score": 5},
        "confidence": 0.8,
    }


def test_strict_verify_fails_when_citation_id_not_in_context(monkeypatch):
    monkeypatch.setattr("nodes.strict_verify.run_prompt", _fake_pass_metrics)

    state = {
        "query": "What is AWS revenue?",
        "response": "AWS revenue is 107,556 million.",
        "citations": [
            {
                "id": "missing-doc-id",
                "title": "Amazon 10K 2024.pdf",
                "quote": "AWS revenue was 107,556 in 2024.",
                "score": 0.9,
            }
        ],
        "context_pool": [
            {
                "id": "doc-1",
                "title": "Amazon 10K 2024.pdf",
                "content": "AWS revenue was 107,556 million in 2024.",
                "score": 1.0,
                "metadata": {"source": "data/10k/Amazon 10K 2024.pdf"},
            }
        ],
        "steps_log": [],
    }

    out = strict_verify_node(state)

    assert out["strict_verdict"] == "FAIL"
    assert out["strict_action"] == "REPAIR"
    assert out["repair_trigger"] == "citation_not_in_context"
    assert out["failure_type"] == "CITATION_MISMATCH"


def test_strict_verify_fails_on_numeric_year_value_mismatch(monkeypatch):
    monkeypatch.setattr("nodes.strict_verify.run_prompt", _fake_pass_metrics)

    state = {
        "query": "In fiscal year 2024, what was AWS revenue?",
        "resolved_query": "In fiscal year 2024, what was AWS revenue?",
        "response": "AWS revenue was 90,757 million in fiscal year 2024.",
        "citations": [
            {
                "id": "doc-1",
                "title": "Amazon 10K 2024.pdf",
                "quote": "AWS revenue was 90,757 million in 2023.",
                "score": 0.9,
            }
        ],
        "context_pool": [
            {
                "id": "doc-1",
                "title": "Amazon 10K 2024.pdf",
                "content": "AWS revenue was 90,757 million in 2023 and 107,556 million in 2024.",
                "score": 1.0,
                "metadata": {"source": "data/10k/Amazon 10K 2024.pdf"},
            }
        ],
        "evidence_table": [
            {
                "company": "Amazon",
                "metric": "AWS net sales",
                "fiscal_year": 2024,
                "value": "107,556",
                "unit": "million",
                "citation_id": "doc-1",
                "citation_title": "Amazon 10K 2024.pdf",
                "quote": "AWS revenue was 90,757 million in 2023 and 107,556 million in 2024.",
                "confidence": 0.95,
            }
        ],
        "steps_log": [],
    }

    out = strict_verify_node(state)

    assert out["strict_verdict"] == "FAIL"
    assert out["strict_action"] == "REPAIR"
    assert out["repair_trigger"] == "numeric_year_value_mismatch"
    assert out["failure_type"] == "LOGICAL_ERROR"


def test_strict_verify_passes_when_numeric_value_matches_target_year(monkeypatch):
    monkeypatch.setattr("nodes.strict_verify.run_prompt", _fake_pass_metrics)

    state = {
        "query": "In fiscal year 2024, what was AWS revenue?",
        "resolved_query": "In fiscal year 2024, what was AWS revenue?",
        "response": "AWS revenue was 107,556 million in fiscal year 2024.",
        "citations": [
            {
                "id": "doc-1",
                "title": "Amazon 10K 2024.pdf",
                "quote": "AWS revenue was 90,757 million in 2023 and 107,556 million in 2024.",
                "score": 0.9,
            }
        ],
        "context_pool": [
            {
                "id": "doc-1",
                "title": "Amazon 10K 2024.pdf",
                "content": "AWS revenue was 90,757 million in 2023 and 107,556 million in 2024.",
                "score": 1.0,
                "metadata": {"source": "data/10k/Amazon 10K 2024.pdf"},
            }
        ],
        "evidence_table": [
            {
                "company": "Amazon",
                "metric": "AWS net sales",
                "fiscal_year": 2024,
                "value": "107,556",
                "unit": "million",
                "citation_id": "doc-1",
                "citation_title": "Amazon 10K 2024.pdf",
                "quote": "AWS revenue was 90,757 million in 2023 and 107,556 million in 2024.",
                "confidence": 0.95,
            }
        ],
        "steps_log": [],
    }

    out = strict_verify_node(state)

    assert out["strict_verdict"] == "PASS"
    assert out["strict_action"] == "PASS"
    assert out.get("failure_type") is None


def test_strict_verify_overrides_llm_logic_contradiction_when_numeric_alignment_is_strong(monkeypatch):
    monkeypatch.setattr("nodes.strict_verify.run_prompt", _fake_contradiction_metrics)

    state = {
        "query": "Compare cloud revenue in fiscal year 2024 for Microsoft, Amazon, and Alphabet.",
        "resolved_query": "Compare cloud revenue in fiscal year 2024 for Microsoft, Amazon, and Alphabet.",
        "response": (
            "Microsoft Cloud revenue was 137.4 billion in fiscal year 2024. "
            "Amazon AWS revenue was 107,556 million in fiscal year 2024. "
            "Alphabet Google Cloud revenue was 43,229 million in fiscal year 2024."
        ),
        "citations": [
            {"id": "ms", "title": "MSFT 10-K.pdf", "quote": "Microsoft Cloud revenue increased 23% to $137.4 billion.", "score": 0.9},
            {"id": "amzn", "title": "Amazon 10K 2024.pdf", "quote": "AWS ... 107,556", "score": 0.9},
            {"id": "goog", "title": "Alphabet 10K 2024.pdf", "quote": "Google Cloud ... 43,229", "score": 0.9},
        ],
        "context_pool": [
            {"id": "ms", "title": "MSFT 10-K.pdf", "content": "Microsoft Cloud revenue increased 23% to $137.4 billion.", "score": 1.0, "metadata": {"source": "data/10k/MSFT 10-K.pdf"}},
            {"id": "amzn", "title": "Amazon 10K 2024.pdf", "content": "AWS ... 107,556", "score": 1.0, "metadata": {"source": "data/10k/Amazon 10K 2024.pdf"}},
            {"id": "goog", "title": "Alphabet 10K 2024.pdf", "content": "Google Cloud ... 43,229", "score": 1.0, "metadata": {"source": "data/10k/Alphabet 10K 2024.pdf"}},
        ],
        "evidence_table": [
            {"company": "Microsoft", "metric": "Microsoft Cloud revenue", "fiscal_year": 2024, "value": "137,400", "unit": "million", "citation_id": "ms", "citation_title": "MSFT 10-K.pdf", "quote": "Microsoft Cloud revenue increased 23% to $137.4 billion.", "confidence": 1.0},
            {"company": "Amazon", "metric": "AWS net sales", "fiscal_year": 2024, "value": "107,556", "unit": "million", "citation_id": "amzn", "citation_title": "Amazon 10K 2024.pdf", "quote": "AWS ... 107,556", "confidence": 1.0},
            {"company": "Alphabet", "metric": "Google Cloud revenue", "fiscal_year": 2024, "value": "43,229", "unit": "million", "citation_id": "goog", "citation_title": "Alphabet 10K 2024.pdf", "quote": "Google Cloud ... 43,229", "confidence": 1.0},
        ],
        "steps_log": [],
    }

    out = strict_verify_node(state)

    assert out["strict_verdict"] == "PASS"
    assert out["strict_action"] == "PASS"
    assert out.get("failure_type") is None
