#!/usr/bin/env python3

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.retrieve_tool.pipeline import RetrievalPipeline


@dataclass
class EvalCase:
    case_id: str
    query: str
    expected_sources: set[str]


def _normalize_source(raw: str) -> str:
    return Path(raw).name.strip().lower()


def _extract_pred_sources(results: list[Any]) -> set[str]:
    sources: set[str] = set()
    for item in results:
        metadata = getattr(item, "metadata", None) or {}
        source = metadata.get("source")
        if source:
            sources.add(_normalize_source(source))
    return sources


def _load_eval_cases(dataset_path: Path) -> list[EvalCase]:
    with dataset_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError("Dataset format invalid: top-level must be a list")

    cases: list[EvalCase] = []
    for i, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset row {i} invalid: must be object")

        case_id = str(row.get("id", f"row-{i}")).strip()
        query = str(row.get("query", "")).strip()
        expected = row.get("expected_sources", [])

        if not query:
            raise ValueError(f"Dataset row {i} invalid: query is empty")
        if not isinstance(expected, list) or not expected:
            raise ValueError(f"Dataset row {i} invalid: expected_sources must be non-empty list")

        cases.append(
            EvalCase(
                case_id=case_id,
                query=query,
                expected_sources={_normalize_source(x) for x in expected},
            )
        )

    return cases


def run_benchmark(dataset_path: Path, output_path: Path | None, verbose: bool, warmup: bool) -> dict[str, Any]:
    cases = _load_eval_cases(dataset_path)
    if not cases:
        raise ValueError("Dataset must contain at least 1 query")

    pipeline = RetrievalPipeline()

    if warmup:
        _ = pipeline.run("warmup query")

    rows: list[dict[str, Any]] = []

    for case in cases:
        start = time.perf_counter()
        results = pipeline.run(case.query)
        latency_ms = (time.perf_counter() - start) * 1000.0

        pred_sources = _extract_pred_sources(results)
        expected_sources = case.expected_sources

        tp = len(pred_sources & expected_sources)
        precision = tp / len(pred_sources) if pred_sources else 0.0
        recall = tp / len(expected_sources) if expected_sources else 1.0

        row = {
            "id": case.case_id,
            "query": case.query,
            "latency_ms": latency_ms,
            "precision": precision,
            "recall": recall,
            "pred_sources": sorted(pred_sources),
            "expected_sources": sorted(expected_sources),
            "hit": sorted(pred_sources & expected_sources),
        }
        rows.append(row)

        if verbose:
            print(
                f"[{case.case_id}] latency={latency_ms:.2f}ms "
                f"precision={precision:.3f} recall={recall:.3f}"
            )

    avg_latency_ms = statistics.mean(x["latency_ms"] for x in rows)
    avg_precision = statistics.mean(x["precision"] for x in rows)
    avg_recall = statistics.mean(x["recall"] for x in rows)
    p50_latency_ms = statistics.median(x["latency_ms"] for x in rows)
    p95_latency_ms = statistics.quantiles(
        [x["latency_ms"] for x in rows],
        n=100,
        method="inclusive",
    )[94]

    summary = {
        "dataset": str(dataset_path),
        "query_count": len(rows),
        "average_latency_ms": avg_latency_ms,
        "p50_latency_ms": p50_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "average_precision": avg_precision,
        "average_recall": avg_recall,
    }

    report = {
        "summary": summary,
        "rows": rows,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run retrieval benchmark and report latency/recall/precision."
    )
    parser.add_argument(
        "--dataset",
        default="data/eval/retrieval_eval_20.json",
        help="Path to evaluation dataset json",
    )
    parser.add_argument(
        "--output",
        default="data/eval/retrieval_benchmark_result.json",
        help="Path to output json report",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Do not write output file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-query metrics",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable warmup query before benchmark",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = None if args.no_output else Path(args.output)

    report = run_benchmark(
        dataset_path=dataset_path,
        output_path=output_path,
        verbose=args.verbose,
        warmup=not args.no_warmup,
    )

    summary = report["summary"]
    print("\n=== Retrieval Benchmark Summary ===")
    print(f"queries: {summary['query_count']}")
    print(f"average latency: {summary['average_latency_ms']:.2f} ms")
    print(f"p50 latency: {summary['p50_latency_ms']:.2f} ms")
    print(f"p95 latency: {summary['p95_latency_ms']:.2f} ms")
    print(f"average precision: {summary['average_precision']:.4f}")
    print(f"average recall: {summary['average_recall']:.4f}")
    if output_path:
        print(f"report saved: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
