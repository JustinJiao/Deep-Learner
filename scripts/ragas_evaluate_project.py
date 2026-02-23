#!/usr/bin/env python3

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_utilization,
    faithfulness,
)

from config.settings import AppConfig
from core.executor import AgentExecutor
from core.registry import NODE_REGISTRY


def _safe_mean(values: list[float]) -> float:
    usable = [x for x in values if not math.isnan(x)]
    if not usable:
        return float("nan")
    return float(sum(usable) / len(usable))


def _normalize_source(raw: str) -> str:
    return Path(raw).name.strip().lower()


def _load_queries(dataset_path: Path, limit: int | None) -> list[dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError("dataset top-level must be a list")

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"row {i} must be an object")

        query = str(item.get("query", "")).strip()
        if not query:
            raise ValueError(f"row {i} query is empty")

        row = {
            "id": str(item.get("id", f"row-{i}")),
            "query": query,
            "ground_truth": str(item.get("ground_truth", "")).strip(),
            "expected_sources": [
                _normalize_source(x) for x in item.get("expected_sources", []) if str(x).strip()
            ],
        }
        rows.append(row)

    if limit is not None:
        rows = rows[:limit]

    if not rows:
        raise ValueError("no valid query rows found")

    return rows


def _collect_contexts(state: dict[str, Any]) -> list[str]:
    contexts: list[str] = []
    for doc in state.get("context_pool", []) or []:
        content = (doc or {}).get("content", "")
        if isinstance(content, str) and content.strip():
            contexts.append(content)
    return contexts


def _collect_sources(state: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for doc in state.get("context_pool", []) or []:
        md = (doc or {}).get("metadata", {}) or {}
        src = md.get("source")
        if src:
            out.append(_normalize_source(str(src)))
    return sorted(set(out))


def _run_executor_dataset(
    rows: list[dict[str, Any]],
    isolate_ltm: bool,
) -> list[dict[str, Any]]:
    executor = AgentExecutor()
    records: list[dict[str, Any]] = []

    original_persist = NODE_REGISTRY.get("persist_ltm")
    original_recall_top_k = AppConfig.LTM_RECALL_TOP_K

    if isolate_ltm:
        NODE_REGISTRY["persist_ltm"] = lambda s: s
        AppConfig.LTM_RECALL_TOP_K = 0

    try:
        for i, row in enumerate(rows, start=1):
            session_id = f"ragas-eval-{int(time.time())}-{i:04d}"
            start = time.perf_counter()
            state = executor.run(session_id=session_id, query=row["query"])
            latency_ms = (time.perf_counter() - start) * 1000.0

            contexts = _collect_contexts(state)
            pred_sources = _collect_sources(state)
            expected_sources = row.get("expected_sources", [])

            tp = len(set(pred_sources) & set(expected_sources))
            retrieval_precision = tp / len(pred_sources) if pred_sources else 0.0
            retrieval_recall = tp / len(expected_sources) if expected_sources else float("nan")

            records.append(
                {
                    "id": row["id"],
                    "question": row["query"],
                    "ground_truth": row.get("ground_truth", ""),
                    "answer": (state.get("response") or "").strip(),
                    "contexts": contexts,
                    "context_count": len(contexts),
                    "pred_sources": pred_sources,
                    "expected_sources": expected_sources,
                    "retrieval_precision": retrieval_precision,
                    "retrieval_recall": retrieval_recall,
                    "run_status": state.get("run_status", "unknown"),
                    "latency_ms": latency_ms,
                }
            )
    finally:
        if isolate_ltm:
            if original_persist is not None:
                NODE_REGISTRY["persist_ltm"] = original_persist
            AppConfig.LTM_RECALL_TOP_K = original_recall_top_k

    return records


def _run_ragas(records: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]], list[str]]:
    has_ground_truth = all((r.get("ground_truth") or "").strip() for r in records)

    data_dict = {
        "question": [r["question"] for r in records],
        "answer": [r["answer"] for r in records],
        "contexts": [r["contexts"] for r in records],
    }
    metric_names: list[str]
    metric_defs: list[Any]

    if has_ground_truth:
        data_dict["ground_truth"] = [r["ground_truth"] for r in records]
        metric_defs = [
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ]
        metric_names = [
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "answer_correctness",
        ]
    else:
        metric_defs = [
            answer_relevancy,
            faithfulness,
            context_utilization,
        ]
        metric_names = [
            "answer_relevancy",
            "faithfulness",
            "context_utilization",
        ]

    hf_ds = Dataset.from_dict(data_dict)

    result = evaluate(
        hf_ds,
        metrics=metric_defs,
    )

    per_row_df = result.to_pandas()
    per_row_metrics: list[dict[str, Any]] = per_row_df.to_dict(orient="records")

    summary_metrics = {
        name: float(result.get(name, float("nan")))
        for name in metric_names
    }
    return summary_metrics, per_row_metrics, metric_names


def run_eval(
    dataset_path: Path,
    output_path: Path | None,
    limit: int | None,
    isolate_ltm: bool,
) -> dict[str, Any]:
    rows = _load_queries(dataset_path, limit)
    records = _run_executor_dataset(rows=rows, isolate_ltm=isolate_ltm)
    ragas_summary, ragas_row_metrics, metric_names = _run_ragas(records)

    merged_rows: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        metric_row = ragas_row_metrics[i] if i < len(ragas_row_metrics) else {}
        merged = dict(rec)
        merged.update(
            {
                "ragas_answer_relevancy": metric_row.get("answer_relevancy"),
                "ragas_context_precision": metric_row.get("context_precision"),
                "ragas_context_recall": metric_row.get("context_recall"),
                "ragas_answer_correctness": metric_row.get("answer_correctness"),
                "ragas_faithfulness": metric_row.get("faithfulness"),
                "ragas_context_utilization": metric_row.get("context_utilization"),
            }
        )
        merged_rows.append(merged)

    avg_latency = statistics.mean(r["latency_ms"] for r in merged_rows)
    p50_latency = statistics.median(r["latency_ms"] for r in merged_rows)
    p95_latency = statistics.quantiles(
        [r["latency_ms"] for r in merged_rows],
        n=100,
        method="inclusive",
    )[94]

    summary = {
        "dataset": str(dataset_path),
        "query_count": len(merged_rows),
        "isolate_ltm": isolate_ltm,
        "average_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "run_status_counts": dict(Counter(r["run_status"] for r in merged_rows)),
        "non_empty_context_rate": (
            sum(1 for r in merged_rows if r["context_count"] > 0) / len(merged_rows)
        ),
        "avg_retrieval_precision": _safe_mean([r["retrieval_precision"] for r in merged_rows]),
        "avg_retrieval_recall": _safe_mean([r["retrieval_recall"] for r in merged_rows]),
        "ragas_metrics": ragas_summary,
        "ragas_metric_names": metric_names,
    }

    report = {
        "summary": summary,
        "rows": merged_rows,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate end-to-end project with RAGAS metrics."
    )
    parser.add_argument(
        "--dataset",
        default="data/eval/retrieval_eval_20.json",
        help="query dataset JSON path",
    )
    parser.add_argument(
        "--output",
        default="data/eval/ragas_project_eval_result.json",
        help="output report JSON path",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="do not write output file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only first N rows",
    )
    parser.add_argument(
        "--no-isolate-ltm",
        action="store_true",
        help="keep original LTM recall/persist behavior during evaluation",
    )
    return parser


def main() -> int:
    load_dotenv(dotenv_path=".env")

    parser = build_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = None if args.no_output else Path(args.output)

    report = run_eval(
        dataset_path=dataset_path,
        output_path=output_path,
        limit=args.limit,
        isolate_ltm=not args.no_isolate_ltm,
    )

    s = report["summary"]
    print("\n=== End-to-End RAGAS Project Eval Summary ===")
    print(f"dataset: {s['dataset']}")
    print(f"queries: {s['query_count']}")
    print(f"isolate_ltm: {s['isolate_ltm']}")
    print(f"average latency: {s['average_latency_ms']:.2f} ms")
    print(f"p50 latency: {s['p50_latency_ms']:.2f} ms")
    print(f"p95 latency: {s['p95_latency_ms']:.2f} ms")
    print(f"run_status_counts: {s['run_status_counts']}")
    print(f"non-empty context rate: {s['non_empty_context_rate']:.4f}")
    print(f"avg retrieval precision: {s['avg_retrieval_precision']:.4f}")
    print(f"avg retrieval recall: {s['avg_retrieval_recall']:.4f}")
    for metric_name, metric_value in s["ragas_metrics"].items():
        print(f"ragas {metric_name}: {metric_value:.4f}")
    if output_path:
        print(f"report saved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
