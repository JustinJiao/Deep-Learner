#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _safe_float(value: Any, fallback: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _fmt(value: Any, digits: int = 4) -> str:
    x = _safe_float(value, fallback=float("nan"))
    if math.isnan(x):
        return "nan"
    return f"{x:.{digits}f}"


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError(f"line {line_no} is not a JSON object")
            rows.append(obj)
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def _sample_by_category(rows: list[dict[str, Any]], per_category: int) -> list[dict[str, Any]]:
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cat = str(row.get("category", "")).strip()
        if not cat:
            raise ValueError("dataset row missing `category`; this benchmark expects categorized rows")
        by_cat[cat].append(row)

    sampled: list[dict[str, Any]] = []
    for category in sorted(by_cat.keys()):
        bucket = by_cat[category]
        if len(bucket) < per_category:
            raise ValueError(
                f"category `{category}` has only {len(bucket)} rows; need at least {per_category}"
            )
        sampled.extend(bucket[:per_category])
    return sampled


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_eval_for_provider(
    project_root: Path,
    python_exec: str,
    dataset_path: Path,
    output_dir: Path,
    run_name: str,
    force_provider: str,
    gemini_api_key: str | None,
    per_query_timeout_seconds: int,
    ragas_timeout_seconds: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    env["LLM_FORCE_PROVIDER"] = force_provider
    env["RUNTIME_V2_ENABLED"] = "true"
    env["RUNTIME_FORCE_RETRIEVE_WHEN_MEMORY_SUFFICIENT"] = "true"
    env["RUNTIME_ENFORCE_CONTRACT"] = "true"
    # 为三家 provider 的可比测试加速：缩小上下文窗口与 rerank 规模，避免单题超时。
    env["RUNTIME_COMPOSE_CONTEXT_TOP_K"] = "8"
    env["RUNTIME_VERIFY_CONTEXT_TOP_K"] = "8"
    env["PHASE1_RERANK_TOP_N"] = "40"
    env["PHASE2_RERANK_TOP_N"] = "60"
    env["OPENAI_TIMEOUT_SECONDS"] = "30"
    env["OPENAI_MAX_RETRIES"] = "0"
    env["ANTHROPIC_TIMEOUT_SECONDS"] = "30"
    env["ANTHROPIC_MAX_RETRIES"] = "0"

    if force_provider == "gemini":
        if gemini_api_key:
            env["GEMINI_API_KEY"] = gemini_api_key
        if not env.get("GEMINI_API_KEY") and not env.get("GOOGLE_API_KEY"):
            raise ValueError("Gemini run requires GEMINI_API_KEY or GOOGLE_API_KEY")
        env["GEMINI_CHAT_MODEL"] = "gemini-2.5-flash"
        env["GEMINI_COMPOSE_MODEL"] = "gemini-2.5-flash"
        env["GEMINI_VERIFY_MODEL"] = "gemini-2.5-flash"
        env["GEMINI_REWRITE_MODEL"] = "gemini-2.5-flash"
        env["GEMINI_MEMORY_MODEL"] = "gemini-2.5-flash"

    cmd = [
        python_exec,
        "evaluation_suite/run_eval.py",
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
        "--k-values",
        "1,3,5",
        "--no-fast-executor",
        "--per-query-timeout-seconds",
        str(int(per_query_timeout_seconds)),
        "--ragas-timeout-seconds",
        str(int(ragas_timeout_seconds)),
    ]
    subprocess.run(cmd, cwd=str(project_root), env=env, check=True)

    summary_path = output_dir / run_name / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _choose_best_provider(rows: list[dict[str, Any]]) -> str:
    def _sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
        correctness = _safe_float(item.get("answer_correctness"), fallback=float("nan"))
        ok_rate = _safe_float(item.get("ok_rate"), fallback=float("nan"))
        latency = _safe_float(item.get("avg_latency_ms"), fallback=float("nan"))
        if math.isnan(correctness):
            correctness = -1.0
        if math.isnan(ok_rate):
            ok_rate = -1.0
        if math.isnan(latency):
            latency = float("inf")
        return (correctness, ok_rate, -latency)

    ranked = sorted(rows, key=_sort_key, reverse=True)
    return str(ranked[0]["provider"]) if ranked else ""


def _choose_best_provider_for_node(node_rows: list[dict[str, Any]]) -> str:
    def _sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
        correctness = _safe_float(item.get("correctness_proxy"), fallback=float("nan"))
        ok_rate = _safe_float(item.get("ok_rate_proxy"), fallback=float("nan"))
        latency = _safe_float(item.get("avg_latency_ms"), fallback=float("nan"))
        if math.isnan(correctness):
            correctness = -1.0
        if math.isnan(ok_rate):
            ok_rate = -1.0
        if math.isnan(latency):
            latency = float("inf")
        return (correctness, ok_rate, -latency)

    ranked = sorted(node_rows, key=_sort_key, reverse=True)
    return str(ranked[0]["provider"]) if ranked else ""


def _build_report_markdown(
    sampled_rows: list[dict[str, Any]],
    provider_summaries: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    category_to_ids: dict[str, list[str]] = defaultdict(list)
    for row in sampled_rows:
        category_to_ids[str(row.get("category", ""))].append(str(row.get("id", "")))

    overall_rows: list[dict[str, Any]] = []
    for provider in ("openai", "claude", "gemini"):
        summary = provider_summaries[provider]
        run_status_counts = summary.get("run_status_counts", {}) or {}
        query_count = max(1, int(summary.get("query_count", 0) or 0))
        ok_rate = _safe_float(run_status_counts.get("ok", 0), fallback=0.0) / query_count
        overall_rows.append(
            {
                "provider": provider,
                "avg_latency_ms": _safe_float(summary.get("average_latency_ms")),
                "p95_latency_ms": _safe_float(summary.get("p95_latency_ms")),
                "answer_correctness": _safe_float((summary.get("ragas", {}) or {}).get("answer_correctness")),
                "answer_relevancy": _safe_float((summary.get("ragas", {}) or {}).get("answer_relevancy")),
                "recall_at_5": _safe_float((summary.get("custom_metrics", {}) or {}).get("recall@5")),
                "precision_at_5": _safe_float((summary.get("custom_metrics", {}) or {}).get("precision@5")),
                "ok_rate": ok_rate,
            }
        )

    node_names: set[str] = set()
    for summary in provider_summaries.values():
        node_names.update((summary.get("llm_node_metrics", {}) or {}).keys())

    node_comparison: dict[str, list[dict[str, Any]]] = {}
    for node in sorted(node_names):
        rows: list[dict[str, Any]] = []
        for provider in ("openai", "claude", "gemini"):
            node_metrics = (provider_summaries[provider].get("llm_node_metrics", {}) or {}).get(node, {}) or {}
            model_map = node_metrics.get("models", {}) or {}
            model_desc = ", ".join(sorted(model_map.keys())) if model_map else ""
            rows.append(
                {
                    "provider": provider,
                    "models": model_desc,
                    "call_count": int(node_metrics.get("call_count", 0) or 0),
                    "avg_latency_ms": _safe_float(node_metrics.get("avg_latency_ms")),
                    "p95_latency_ms": _safe_float(node_metrics.get("p95_latency_ms")),
                    "correctness_proxy": _safe_float(node_metrics.get("avg_answer_correctness_proxy")),
                    "relevancy_proxy": _safe_float(node_metrics.get("avg_answer_relevancy_proxy")),
                    "ok_rate_proxy": _safe_float(node_metrics.get("ok_rate_proxy")),
                    "error_count": int(node_metrics.get("error_count", 0) or 0),
                }
            )
        node_comparison[node] = rows

    best_overall_provider = _choose_best_provider(overall_rows)
    best_provider_by_node = {
        node: _choose_best_provider_for_node(rows)
        for node, rows in node_comparison.items()
    }

    lines: list[str] = []
    lines.append("# Provider Benchmark Summary")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Query sample count: {len(sampled_rows)}")
    lines.append("- Sampling rule: first 3 questions per category from `testset_10k_3cats_60.jsonl`")
    lines.append("")
    lines.append("## Sampled Queries")
    for category in sorted(category_to_ids.keys()):
        lines.append(f"- {category}: {', '.join(category_to_ids[category])}")
    lines.append("")
    lines.append("## Overall Comparison")
    lines.append("| Provider | Avg Latency(ms) | P95 Latency(ms) | Answer Correctness | Answer Relevancy | Recall@5 | Precision@5 | OK Rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in overall_rows:
        lines.append(
            f"| {row['provider']} | {_fmt(row['avg_latency_ms'], 2)} | {_fmt(row['p95_latency_ms'], 2)} | "
            f"{_fmt(row['answer_correctness'])} | {_fmt(row['answer_relevancy'])} | "
            f"{_fmt(row['recall_at_5'])} | {_fmt(row['precision_at_5'])} | {_fmt(row['ok_rate'])} |"
        )
    lines.append("")
    lines.append(f"- Best overall provider (quality-first): **{best_overall_provider}**")
    lines.append("")
    lines.append("## Node-Level Comparison (LLM Nodes)")
    for node in sorted(node_comparison.keys()):
        lines.append(f"### `{node}`")
        lines.append("| Provider | Model(s) | Calls | Avg Latency(ms) | P95 Latency(ms) | Correctness Proxy | Relevancy Proxy | OK Proxy | Errors |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in node_comparison[node]:
            lines.append(
                f"| {row['provider']} | {row['models']} | {row['call_count']} | {_fmt(row['avg_latency_ms'], 2)} | "
                f"{_fmt(row['p95_latency_ms'], 2)} | {_fmt(row['correctness_proxy'])} | "
                f"{_fmt(row['relevancy_proxy'])} | {_fmt(row['ok_rate_proxy'])} | {row['error_count']} |"
            )
        lines.append(f"- Best provider for this node (quality-first): **{best_provider_by_node[node]}**")
        lines.append("")

    report_json = {
        "sampled_ids_by_category": dict(category_to_ids),
        "overall": overall_rows,
        "best_overall_provider": best_overall_provider,
        "nodes": node_comparison,
        "best_provider_by_node": best_provider_by_node,
    }
    return "\n".join(lines).strip() + "\n", report_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark all LLM nodes with OpenAI / Claude / Gemini on 3x3 sampled queries.",
    )
    parser.add_argument(
        "--dataset",
        default="evaluation_suite/datasets/testset_10k_3cats_60.jsonl",
        help="Source dataset with `category` field.",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=3,
        help="How many queries to sample per category.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_suite/outputs",
        help="Base output directory.",
    )
    parser.add_argument(
        "--run-name",
        default=f"provider_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Subdirectory under output-dir.",
    )
    parser.add_argument(
        "--python-exec",
        default=".venv/bin/python",
        help="Python executable used to invoke run_eval.py.",
    )
    parser.add_argument(
        "--gemini-api-key",
        default="",
        help="Optional Gemini API key (otherwise read from GEMINI_API_KEY/GOOGLE_API_KEY env).",
    )
    parser.add_argument(
        "--per-query-timeout-seconds",
        type=int,
        default=0,
        help="Hard timeout for each query run in run_eval.py. Use 0 to disable subprocess timeout.",
    )
    parser.add_argument(
        "--ragas-timeout-seconds",
        type=int,
        default=240,
        help="Hard timeout for RAGAS stage in run_eval.py.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = (project_root / args.dataset).resolve()
    output_root = (project_root / args.output_dir / args.run_name).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl_rows(dataset_path)
    sampled_rows = _sample_by_category(rows, per_category=max(1, int(args.per_category)))
    sampled_dataset_path = output_root / "sampled_3x3.jsonl"
    _write_jsonl(sampled_dataset_path, sampled_rows)

    provider_summaries: dict[str, dict[str, Any]] = {}
    provider_specs = [
        ("openai", "openai"),
        ("claude", "anthropic"),
        ("gemini", "gemini"),
    ]
    for provider_label, force_provider in provider_specs:
        run_name = f"{provider_label}_full_chain"
        print(f"\n=== Running provider: {provider_label} ===", flush=True)
        summary = _run_eval_for_provider(
            project_root=project_root,
            python_exec=args.python_exec,
            dataset_path=sampled_dataset_path,
            output_dir=output_root,
            run_name=run_name,
            force_provider=force_provider,
            gemini_api_key=args.gemini_api_key.strip() or None,
            per_query_timeout_seconds=int(args.per_query_timeout_seconds),
            ragas_timeout_seconds=int(args.ragas_timeout_seconds),
        )
        provider_summaries[provider_label] = summary
        print(
            f"[{provider_label}] avg_latency_ms={_fmt(summary.get('average_latency_ms'), 2)} "
            f"answer_correctness={_fmt((summary.get('ragas', {}) or {}).get('answer_correctness'))}",
            flush=True,
        )

    report_md, report_json = _build_report_markdown(sampled_rows, provider_summaries)
    report_md_path = output_root / "provider_benchmark_report.md"
    report_json_path = output_root / "provider_benchmark_report.json"
    report_md_path.write_text(report_md, encoding="utf-8")
    with report_json_path.open("w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)

    print(f"\noutputs: {output_root}", flush=True)
    print(f"report: {report_md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
