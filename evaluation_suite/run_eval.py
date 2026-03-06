#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import statistics
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from beir.retrieval.evaluation import EvaluateRetrieval
from datasets import Dataset, Features, Sequence, Value, load_dataset, load_from_disk
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from config.settings import AppConfig
from core.executor import AgentExecutor
from core.plan import ExecutionPlan
from core.registry import NODE_REGISTRY

REQUIRED_FIELDS = ("question", "ground_truth_answer", "ground_truth_docs")


def _normalize_doc_id(raw: Any) -> str:
    text = str(raw or "").strip()
    return Path(text).name.lower()


def _safe_mean(values: list[float]) -> float:
    valid = [x for x in values if not math.isnan(x)]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_p95(values: list[float]) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    return float(statistics.quantiles(values, n=100, method="inclusive")[94])


def _empty_ragas_summary() -> dict[str, float]:
    return {
        "answer_relevancy": float("nan"),
        "context_precision": float("nan"),
        "context_recall": float("nan"),
        "answer_correctness": float("nan"),
    }


def _trim_contexts_for_ragas(
    contexts: list[str],
    context_top_k: int,
    context_max_chars: int,
) -> list[str]:
    picked = contexts if context_top_k <= 0 else contexts[:context_top_k]
    if context_max_chars <= 0:
        return [str(c or "") for c in picked]
    return [str(c or "")[:context_max_chars] for c in picked]


def _parse_k_values(raw: str) -> list[int]:
    items = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"k must be > 0, got {value}")
        items.append(value)
    if not items:
        raise ValueError("k-values cannot be empty")
    return sorted(set(items))


def _parse_ground_truth_docs(raw: Any) -> list[str]:
    if isinstance(raw, list):
        docs = [_normalize_doc_id(x) for x in raw if str(x).strip()]
    elif isinstance(raw, str):
        docs = [_normalize_doc_id(x) for x in raw.split(",") if x.strip()]
    else:
        docs = []
    return sorted(set(docs))


def load_testset(dataset_path: Path, limit: int | None) -> list[dict[str, Any]]:
    # 支持两种输入：
    # 1) json/jsonl 文件
    # 2) HuggingFace save_to_disk 目录（含 state.json）
    if dataset_path.is_dir() and (dataset_path / "state.json").exists():
        hf_dataset = load_from_disk(str(dataset_path))
    else:
        hf_dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    if limit is not None:
        hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(hf_dataset, start=1):
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"row {idx} missing required fields: {missing}")

        question = str(row.get("question", "")).strip()
        gt_answer = str(row.get("ground_truth_answer", "")).strip()
        gt_docs = _parse_ground_truth_docs(row.get("ground_truth_docs"))
        row_id = str(row.get("id", f"q{idx:04d}")).strip()

        if not question:
            raise ValueError(f"row {idx} question is empty")
        if not gt_answer:
            raise ValueError(f"row {idx} ground_truth_answer is empty")
        if not gt_docs:
            raise ValueError(f"row {idx} ground_truth_docs is empty")

        rows.append(
            {
                "id": row_id,
                "question": question,
                "ground_truth_answer": gt_answer,
                "ground_truth_docs": gt_docs,
            }
        )

    if not rows:
        raise ValueError("dataset has no valid rows")
    return rows


def _extract_retrieval(state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    context_pool = state.get("context_pool", []) or []
    contexts: list[str] = []
    docs: list[dict[str, Any]] = []

    for rank, item in enumerate(context_pool, start=1):
        item = item or {}
        content = str(item.get("content", "") or "").strip()
        if content:
            contexts.append(content)

        metadata = item.get("metadata", {}) or {}
        source = metadata.get("source") or item.get("title") or item.get("id")
        doc_id = _normalize_doc_id(source)
        if not doc_id:
            continue

        docs.append(
            {
                "doc_id": doc_id,
                "rank": rank,
                "score": _safe_float(item.get("score"), fallback=float(len(context_pool) - rank + 1)),
            }
        )

    # 按首出现顺序去重，保留最高分
    ordered_doc_ids: list[str] = []
    score_map: dict[str, float] = {}
    for doc in docs:
        doc_id = doc["doc_id"]
        if doc_id not in ordered_doc_ids:
            ordered_doc_ids.append(doc_id)
            score_map[doc_id] = doc["score"]
        else:
            score_map[doc_id] = max(score_map[doc_id], doc["score"])

    dedup_docs = [
        {
            "doc_id": doc_id,
            "score": score_map[doc_id],
            "rank": i + 1,
        }
        for i, doc_id in enumerate(ordered_doc_ids)
    ]
    return dedup_docs, contexts


def _recall_at_k(gt_docs: list[str], pred_docs: list[str], k: int) -> float:
    if not gt_docs:
        return float("nan")
    gt = set(gt_docs)
    topk = set(pred_docs[:k])
    return len(gt & topk) / len(gt)


def _precision_at_k(gt_docs: list[str], pred_docs: list[str], k: int) -> float:
    topk = pred_docs[:k]
    if not topk:
        return 0.0
    gt = set(gt_docs)
    return len(gt & set(topk)) / len(topk)


def _extract_runtime_signals(state: dict[str, Any]) -> dict[str, Any]:
    strict_fail_types: list[str] = []
    for log in state.get("steps_log", []) or []:
        if isinstance(log, dict):
            node = log.get("node")
            info = log.get("info", {}) or {}
        else:
            node = getattr(log, "node", None)
            info = getattr(log, "info", {}) or {}
        if node != "strict_verify":
            continue
        if not isinstance(info, dict):
            continue
        llm_output = info.get("llm_output", {}) or {}
        if not isinstance(llm_output, dict):
            continue
        verdict = str(llm_output.get("verdict", "")).strip().upper()
        failure_type = str(llm_output.get("failure_type", "")).strip().upper()
        if verdict == "FAIL" and failure_type:
            strict_fail_types.append(failure_type)

    return {
        "runtime_stage": state.get("runtime_stage"),
        "memory_verdict": state.get("memory_verdict"),
        "phase2_used": bool(state.get("phase2_used", False)),
        "repair_used": bool(state.get("repair_used", False)),
        "strict_verdict": state.get("strict_verdict"),
        "failure_type": state.get("failure_type"),
        "strict_fail_types": strict_fail_types,
    }


def _executor_worker(
    query: str,
    session_id: str,
    isolate_ltm: bool,
    fast_executor: bool,
    out_q: mp.Queue,
) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from config.settings import AppConfig as WorkerAppConfig
    from core.executor import AgentExecutor as WorkerAgentExecutor
    from core.plan import ExecutionPlan as WorkerExecutionPlan
    from core.registry import NODE_REGISTRY as WORKER_NODE_REGISTRY

    original_persist = WORKER_NODE_REGISTRY.get("persist_ltm")
    original_intent = WORKER_NODE_REGISTRY.get("intent")
    original_planner = WORKER_NODE_REGISTRY.get("planner")
    original_ltm_top_k = WorkerAppConfig.LTM_RECALL_TOP_K

    try:
        if isolate_ltm:
            WORKER_NODE_REGISTRY["persist_ltm"] = lambda s: s
            WorkerAppConfig.LTM_RECALL_TOP_K = 0

        if fast_executor:
            def _fast_intent_node(state: dict[str, Any]) -> dict[str, Any]:
                state["intent"] = {"type": "research", "confidence": 1.0}
                return state

            def _fast_planner_node(state: dict[str, Any]) -> dict[str, Any]:
                state["is_direct_path"] = False
                # ComposePrompt 的 READS 字段在快路径中显式补齐，避免 PromptContractError
                state.setdefault("rewritten_query", state.get("query", ""))
                state.setdefault("long_term_memory", "无相关长期记忆")
                state.setdefault("short_term_memory", "")
                state.setdefault("recent_messages", [])
                state.setdefault("repair_hint", "")
                state["plan"] = WorkerExecutionPlan(steps=["retrieve", "compose"], max_loops=0)
                return state

            WORKER_NODE_REGISTRY["intent"] = _fast_intent_node
            WORKER_NODE_REGISTRY["planner"] = _fast_planner_node

        state = WorkerAgentExecutor().run(session_id=session_id, query=query)
        runtime_signals = _extract_runtime_signals(state)
        out_q.put(
            {
                "run_status": state.get("run_status", "unknown"),
                "response": state.get("response", ""),
                "context_pool": state.get("context_pool", []) or [],
                "error": state.get("error"),
                **runtime_signals,
            }
        )
    except Exception as e:
        out_q.put(
            {
                "run_status": "error",
                "response": f"evaluation runtime error: {type(e).__name__}: {e}",
                "context_pool": [],
                "error": {"type": type(e).__name__, "message": str(e)},
                "runtime_stage": "",
                "memory_verdict": "",
                "phase2_used": False,
                "repair_used": False,
                "strict_verdict": "",
                "failure_type": "",
                "strict_fail_types": [],
            }
        )
    finally:
        if isolate_ltm:
            if original_persist is not None:
                WORKER_NODE_REGISTRY["persist_ltm"] = original_persist
            WorkerAppConfig.LTM_RECALL_TOP_K = original_ltm_top_k
        if fast_executor:
            if original_intent is not None:
                WORKER_NODE_REGISTRY["intent"] = original_intent
            if original_planner is not None:
                WORKER_NODE_REGISTRY["planner"] = original_planner


def _run_single_query_state_inline(
    query: str,
    session_id: str,
    isolate_ltm: bool,
    fast_executor: bool,
) -> dict[str, Any]:
    original_persist = NODE_REGISTRY.get("persist_ltm")
    original_intent = NODE_REGISTRY.get("intent")
    original_planner = NODE_REGISTRY.get("planner")
    original_ltm_top_k = AppConfig.LTM_RECALL_TOP_K

    try:
        if isolate_ltm:
            NODE_REGISTRY["persist_ltm"] = lambda s: s
            AppConfig.LTM_RECALL_TOP_K = 0

        if fast_executor:
            def _fast_intent_node(state: dict[str, Any]) -> dict[str, Any]:
                state["intent"] = {"type": "research", "confidence": 1.0}
                return state

            def _fast_planner_node(state: dict[str, Any]) -> dict[str, Any]:
                state["is_direct_path"] = False
                # ComposePrompt 的 READS 字段在快路径中显式补齐，避免 PromptContractError
                state.setdefault("rewritten_query", state.get("query", ""))
                state.setdefault("long_term_memory", "无相关长期记忆")
                state.setdefault("short_term_memory", "")
                state.setdefault("recent_messages", [])
                state.setdefault("repair_hint", "")
                state["plan"] = ExecutionPlan(steps=["retrieve", "compose"], max_loops=0)
                return state

            NODE_REGISTRY["intent"] = _fast_intent_node
            NODE_REGISTRY["planner"] = _fast_planner_node

        state = AgentExecutor().run(session_id=session_id, query=query)
        runtime_signals = _extract_runtime_signals(state)
        return {
            "run_status": state.get("run_status", "unknown"),
            "response": state.get("response", ""),
            "context_pool": state.get("context_pool", []) or [],
            "error": state.get("error"),
            **runtime_signals,
        }
    except Exception as e:
        return {
            "run_status": "error",
            "response": f"evaluation runtime error: {type(e).__name__}: {e}",
            "context_pool": [],
            "error": {"type": type(e).__name__, "message": str(e)},
            "runtime_stage": "",
            "memory_verdict": "",
            "phase2_used": False,
            "repair_used": False,
            "strict_verdict": "",
            "failure_type": "",
            "strict_fail_types": [],
        }
    finally:
        if isolate_ltm:
            if original_persist is not None:
                NODE_REGISTRY["persist_ltm"] = original_persist
            AppConfig.LTM_RECALL_TOP_K = original_ltm_top_k
        if fast_executor:
            if original_intent is not None:
                NODE_REGISTRY["intent"] = original_intent
            if original_planner is not None:
                NODE_REGISTRY["planner"] = original_planner


def _run_single_query_state(
    query: str,
    session_id: str,
    isolate_ltm: bool,
    fast_executor: bool,
    per_query_timeout_seconds: int | None,
) -> dict[str, Any]:
    # 当未启用硬超时时，直接在当前进程执行，避免每条 query 的进程启动开销。
    if per_query_timeout_seconds is None:
        return _run_single_query_state_inline(
            query=query,
            session_id=session_id,
            isolate_ltm=isolate_ltm,
            fast_executor=fast_executor,
        )

    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_executor_worker,
        args=(query, session_id, isolate_ltm, fast_executor, out_q),
    )
    proc.start()
    proc.join(timeout=float(per_query_timeout_seconds) if per_query_timeout_seconds else None)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "run_status": "timeout",
            "response": "",
            "context_pool": [],
            "error": {"type": "TimeoutError", "message": "per-query timeout exceeded"},
            "runtime_stage": "",
            "memory_verdict": "",
            "phase2_used": False,
            "repair_used": False,
            "strict_verdict": "",
            "failure_type": "",
            "strict_fail_types": [],
        }

    if out_q.empty():
        return {
            "run_status": "error",
            "response": "evaluation runtime error: child process returned no result",
            "context_pool": [],
            "error": {"type": "RuntimeError", "message": "child process returned no result"},
            "runtime_stage": "",
            "memory_verdict": "",
            "phase2_used": False,
            "repair_used": False,
            "strict_verdict": "",
            "failure_type": "",
            "strict_fail_types": [],
        }

    return out_q.get()


def _run_executor(
    rows: list[dict[str, Any]],
    k_values: list[int],
    isolate_ltm: bool,
    session_prefix: str,
    per_query_timeout_seconds: int | None,
    fast_executor: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        session_id = f"{session_prefix}-{idx:04d}"
        start = time.perf_counter()
        state = _run_single_query_state(
            query=row["question"],
            session_id=session_id,
            isolate_ltm=isolate_ltm,
            fast_executor=fast_executor,
            per_query_timeout_seconds=per_query_timeout_seconds,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        retrieved_docs, contexts = _extract_retrieval(state)
        pred_doc_ids = [doc["doc_id"] for doc in retrieved_docs]

        recall_at_k = {f"recall@{k}": _recall_at_k(row["ground_truth_docs"], pred_doc_ids, k) for k in k_values}
        precision_at_k = {
            f"precision@{k}": _precision_at_k(row["ground_truth_docs"], pred_doc_ids, k) for k in k_values
        }

        records.append(
            {
                "id": row["id"],
                "question": row["question"],
                "ground_truth_answer": row["ground_truth_answer"],
                "ground_truth_docs": row["ground_truth_docs"],
                "run_status": state.get("run_status", "unknown"),
                "latency_ms": latency_ms,
                "answer": str(state.get("response") or "").strip(),
                "retrieved_docs": retrieved_docs,
                "contexts": contexts,
                "context_count": len(contexts),
                "recall_at_k": recall_at_k,
                "precision_at_k": precision_at_k,
                "error_detail": state.get("error"),
                "runtime_stage": state.get("runtime_stage", ""),
                "memory_verdict": state.get("memory_verdict", ""),
                "phase2_used": bool(state.get("phase2_used", False)),
                "repair_used": bool(state.get("repair_used", False)),
                "strict_verdict": state.get("strict_verdict", ""),
                "failure_type": state.get("failure_type", ""),
                "strict_fail_types": state.get("strict_fail_types", []) or [],
            }
        )
        print(
            f"[executor] {idx}/{len(rows)} status={records[-1]['run_status']} "
            f"latency_ms={records[-1]['latency_ms']:.2f}",
            flush=True,
        )

    return records


def _run_ragas_inline(
    records: list[dict[str, Any]],
    context_top_k: int,
    context_max_chars: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    ragas_features = Features(
        {
            "question": Value("string"),
            "answer": Value("string"),
            "contexts": Sequence(Value("string")),
            "ground_truth": Value("string"),
        }
    )
    ragas_ds = Dataset.from_dict(
        {
            "question": [r["question"] for r in records],
            "answer": [r["answer"] for r in records],
            "contexts": [
                _trim_contexts_for_ragas(
                    contexts=r.get("contexts", []) or [],
                    context_top_k=context_top_k,
                    context_max_chars=context_max_chars,
                )
                for r in records
            ],
            "ground_truth": [r["ground_truth_answer"] for r in records],
        },
        features=ragas_features,
    )

    result = evaluate(
        ragas_ds,
        metrics=[answer_relevancy, context_precision, context_recall, answer_correctness],
    )
    row_metrics = result.to_pandas().to_dict(orient="records")
    summary = {
        "answer_relevancy": float(result.get("answer_relevancy", float("nan"))),
        "context_precision": float(result.get("context_precision", float("nan"))),
        "context_recall": float(result.get("context_recall", float("nan"))),
        "answer_correctness": float(result.get("answer_correctness", float("nan"))),
    }
    return summary, row_metrics


def _ragas_worker(
    records: list[dict[str, Any]],
    context_top_k: int,
    context_max_chars: int,
    out_q: mp.Queue,
) -> None:
    try:
        summary, rows = _run_ragas_inline(
            records=records,
            context_top_k=context_top_k,
            context_max_chars=context_max_chars,
        )
        out_q.put(
            {
                "ok": True,
                "summary": summary,
                "rows": rows,
            }
        )
    except Exception as e:
        out_q.put(
            {
                "ok": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
            }
        )


def _run_ragas(
    records: list[dict[str, Any]],
    ragas_timeout_seconds: int | None,
    ragas_context_top_k: int,
    ragas_context_max_chars: int,
    enable_ragas: bool,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    if not enable_ragas:
        return _empty_ragas_summary(), [], {"status": "disabled", "error": None}

    if ragas_timeout_seconds is None:
        try:
            summary, rows = _run_ragas_inline(
                records=records,
                context_top_k=ragas_context_top_k,
                context_max_chars=ragas_context_max_chars,
            )
            return summary, rows, {"status": "ok", "error": None}
        except Exception as e:
            return _empty_ragas_summary(), [], {
                "status": "error",
                "error": {"type": type(e).__name__, "message": str(e)},
            }

    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_ragas_worker,
        args=(records, ragas_context_top_k, ragas_context_max_chars, out_q),
    )
    proc.start()
    proc.join(timeout=float(ragas_timeout_seconds))

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return _empty_ragas_summary(), [], {
            "status": "timeout",
            "error": {"type": "TimeoutError", "message": "ragas timeout exceeded"},
        }

    if out_q.empty():
        return _empty_ragas_summary(), [], {
            "status": "error",
            "error": {"type": "RuntimeError", "message": "ragas worker returned no result"},
        }

    payload = out_q.get()
    if bool(payload.get("ok", False)):
        return payload["summary"], payload["rows"], {"status": "ok", "error": None}

    return _empty_ragas_summary(), [], {
        "status": "error",
        "error": payload.get("error"),
    }


def _run_beir(records: list[dict[str, Any]], k_values: list[int]) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, int]] = {}
    results: dict[str, dict[str, float]] = {}

    for record in records:
        query_id = record["id"]
        qrels[query_id] = {doc_id: 1 for doc_id in record["ground_truth_docs"]}
        results[query_id] = {
            doc["doc_id"]: _safe_float(doc.get("score"), fallback=0.0)
            for doc in record["retrieved_docs"]
        }

    ndcg, map_, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)
    return {
        "ndcg": ndcg,
        "map": map_,
        "recall": recall,
        "precision": precision,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _flatten_for_csv(records: list[dict[str, Any]], k_values: list[int]) -> list[dict[str, Any]]:
    flat_rows: list[dict[str, Any]] = []
    for row in records:
        out = {
            "id": row["id"],
            "question": row["question"],
            "run_status": row["run_status"],
            "latency_ms": round(row["latency_ms"], 3),
            "context_count": row["context_count"],
            "ground_truth_docs": "|".join(row["ground_truth_docs"]),
            "retrieved_doc_ids": "|".join(doc["doc_id"] for doc in row["retrieved_docs"]),
            "answer": row["answer"],
            "ragas_answer_relevancy": row.get("ragas_answer_relevancy"),
            "ragas_context_precision": row.get("ragas_context_precision"),
            "ragas_context_recall": row.get("ragas_context_recall"),
            "ragas_answer_correctness": row.get("ragas_answer_correctness"),
            "runtime_stage": row.get("runtime_stage", ""),
            "memory_verdict": row.get("memory_verdict", ""),
            "phase2_used": row.get("phase2_used", False),
            "repair_used": row.get("repair_used", False),
            "strict_verdict": row.get("strict_verdict", ""),
            "failure_type": row.get("failure_type", ""),
            "strict_fail_types": "|".join(row.get("strict_fail_types", []) or []),
        }
        for k in k_values:
            out[f"recall@{k}"] = row["recall_at_k"][f"recall@{k}"]
            out[f"precision@{k}"] = row["precision_at_k"][f"precision@{k}"]
        flat_rows.append(out)
    return flat_rows


def _render_stats_table(summary: dict[str, Any]) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    lines.append(f"| query_count | {summary['query_count']} |")
    lines.append(f"| average_latency_ms | {summary['average_latency_ms']:.2f} |")
    lines.append(f"| p50_latency_ms | {summary['p50_latency_ms']:.2f} |")
    lines.append(f"| p95_latency_ms | {summary['p95_latency_ms']:.2f} |")
    lines.append(f"| non_empty_context_rate | {summary['non_empty_context_rate']:.4f} |")
    lines.append(f"| run_status_counts | {summary['run_status_counts']} |")
    lines.append(f"| ragas_status | {summary.get('ragas_status', 'unknown')} |")
    lines.append(f"| memory_sufficient_rate | {summary['memory_sufficient_rate']:.4f} |")
    lines.append(f"| phase2_trigger_rate | {summary['phase2_trigger_rate']:.4f} |")
    lines.append(f"| repair_trigger_rate | {summary['repair_trigger_rate']:.4f} |")
    lines.append(f"| strict_fail_breakdown | {summary['strict_fail_breakdown']} |")

    for key, value in summary["custom_metrics"].items():
        lines.append(f"| {key} | {value:.4f} |")
    for key, value in summary["ragas"].items():
        lines.append(f"| ragas_{key} | {value:.4f} |")
    for key, value in summary["beir_precision"].items():
        lines.append(f"| beir_{key} | {value:.4f} |")
    return "\n".join(lines) + "\n"


def run_all(
    dataset_path: Path,
    output_dir: Path,
    run_name: str,
    k_values: list[int],
    limit: int | None,
    isolate_ltm: bool,
    per_query_timeout_seconds: int | None,
    fast_executor: bool,
    ragas_timeout_seconds: int | None,
    ragas_context_top_k: int,
    ragas_context_max_chars: int,
    enable_ragas: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = load_testset(dataset_path=dataset_path, limit=limit)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_prefix = f"{run_name}-{timestamp}"
    records = _run_executor(
        rows=rows,
        k_values=k_values,
        isolate_ltm=isolate_ltm,
        session_prefix=session_prefix,
        per_query_timeout_seconds=per_query_timeout_seconds,
        fast_executor=fast_executor,
    )
    ragas_summary, ragas_rows, ragas_meta = _run_ragas(
        records=records,
        ragas_timeout_seconds=ragas_timeout_seconds,
        ragas_context_top_k=ragas_context_top_k,
        ragas_context_max_chars=ragas_context_max_chars,
        enable_ragas=enable_ragas,
    )
    beir_metrics = _run_beir(records, k_values)

    if ragas_rows and len(ragas_rows) == len(records):
        for rec, ragas in zip(records, ragas_rows):
            rec["ragas_answer_relevancy"] = ragas.get("answer_relevancy")
            rec["ragas_context_precision"] = ragas.get("context_precision")
            rec["ragas_context_recall"] = ragas.get("context_recall")
            rec["ragas_answer_correctness"] = ragas.get("answer_correctness")
    else:
        for rec in records:
            rec["ragas_answer_relevancy"] = float("nan")
            rec["ragas_context_precision"] = float("nan")
            rec["ragas_context_recall"] = float("nan")
            rec["ragas_answer_correctness"] = float("nan")

    custom_scores: dict[str, float] = {}
    for k in k_values:
        custom_scores[f"recall@{k}"] = _safe_mean([r["recall_at_k"][f"recall@{k}"] for r in records])
        custom_scores[f"precision@{k}"] = _safe_mean([r["precision_at_k"][f"precision@{k}"] for r in records])

    precision_from_beir = {
        key: float(value)
        for key, value in beir_metrics["precision"].items()
    }

    strict_fail_counter: Counter[str] = Counter()
    for record in records:
        for failure_type in record.get("strict_fail_types", []) or []:
            ft = str(failure_type).strip().upper()
            if ft:
                strict_fail_counter[ft] += 1

    summary = {
        "dataset_path": str(dataset_path),
        "query_count": len(records),
        "k_values": k_values,
        "isolate_ltm": isolate_ltm,
        "runtime_v2_enabled": bool(AppConfig.RUNTIME_V2_ENABLED),
        "fast_executor": fast_executor,
        "per_query_timeout_seconds": 0 if per_query_timeout_seconds is None else int(per_query_timeout_seconds),
        "ragas_enabled": bool(enable_ragas),
        "ragas_timeout_seconds": 0 if ragas_timeout_seconds is None else int(ragas_timeout_seconds),
        "ragas_context_top_k": int(ragas_context_top_k),
        "ragas_context_max_chars": int(ragas_context_max_chars),
        "ragas_status": ragas_meta.get("status", "unknown"),
        "ragas_error": ragas_meta.get("error"),
        "average_latency_ms": statistics.mean(r["latency_ms"] for r in records),
        "p50_latency_ms": statistics.median(r["latency_ms"] for r in records),
        "p95_latency_ms": _safe_p95([r["latency_ms"] for r in records]),
        "non_empty_context_rate": sum(1 for r in records if r["context_count"] > 0) / len(records),
        "runtime_stage_counts": dict(Counter(str(r.get("runtime_stage", "")) for r in records)),
        "memory_sufficient_rate": (
            sum(1 for r in records if str(r.get("memory_verdict", "")).upper() == "SUFFICIENT") / len(records)
        ),
        "phase2_trigger_rate": (
            sum(1 for r in records if bool(r.get("phase2_used", False))) / len(records)
        ),
        "repair_trigger_rate": (
            sum(1 for r in records if bool(r.get("repair_used", False))) / len(records)
        ),
        "strict_fail_breakdown": dict(strict_fail_counter),
        "run_status_counts": dict(Counter(r["run_status"] for r in records)),
        "custom_metrics": custom_scores,
        "ragas": ragas_summary,
        "beir_precision": precision_from_beir,
        "beir_raw": beir_metrics,
    }

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    records_jsonl_path = run_dir / "records.jsonl"
    records_csv_path = run_dir / "records.csv"
    stats_md_path = run_dir / "stats_table.md"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _write_jsonl(records_jsonl_path, records)

    flat_rows = _flatten_for_csv(records, k_values)
    with records_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    stats_md_path.write_text(_render_stats_table(summary), encoding="utf-8")
    return summary, records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation: executor + latency + retrieval@k + RAGAS + BEIR.",
    )
    parser.add_argument(
        "--dataset",
        default="evaluation_suite/datasets/testset_20.jsonl",
        help="Path to dataset (json/jsonl) with fields: question, ground_truth_answer, ground_truth_docs",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_suite/outputs",
        help="Directory for run outputs",
    )
    parser.add_argument(
        "--run-name",
        default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Subdirectory name under output-dir",
    )
    parser.add_argument(
        "--k-values",
        default="1,3,5",
        help="Comma-separated top-k values for custom recall@k/precision@k and BEIR metrics",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate first N rows",
    )
    parser.add_argument(
        "--no-isolate-ltm",
        action="store_true",
        help="Use production LTM read/write during eval (default is isolated)",
    )
    parser.add_argument(
        "--per-query-timeout-seconds",
        type=int,
        default=0,
        help="Hard timeout per executor query. Default 0 (disabled, no subprocess spawn overhead).",
    )
    parser.add_argument(
        "--fast-executor",
        dest="fast_executor",
        action="store_true",
        default=True,
        help="Use fast intent/planner path in legacy runtime. Enabled by default.",
    )
    parser.add_argument(
        "--no-fast-executor",
        dest="fast_executor",
        action="store_false",
        help="Disable fast path and run full chain in legacy runtime.",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Disable RAGAS stage (executor + retrieval metrics only).",
    )
    parser.add_argument(
        "--ragas-timeout-seconds",
        type=int,
        default=240,
        help="Timeout for RAGAS stage. Default 240s. Set <=0 to disable timeout.",
    )
    parser.add_argument(
        "--ragas-context-top-k",
        type=int,
        default=8,
        help="Only pass top-k contexts per sample to RAGAS. Default 8.",
    )
    parser.add_argument(
        "--ragas-context-max-chars",
        type=int,
        default=1200,
        help="Max chars per context chunk passed to RAGAS. Default 1200.",
    )
    return parser


def main() -> int:
    load_dotenv(dotenv_path=".env")

    args = build_parser().parse_args()
    k_values = _parse_k_values(args.k_values)
    summary, _records = run_all(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        run_name=args.run_name,
        k_values=k_values,
        limit=args.limit,
        isolate_ltm=not args.no_isolate_ltm,
        per_query_timeout_seconds=(
            None if args.per_query_timeout_seconds <= 0 else args.per_query_timeout_seconds
        ),
        fast_executor=args.fast_executor,
        ragas_timeout_seconds=(
            None if args.ragas_timeout_seconds <= 0 else args.ragas_timeout_seconds
        ),
        ragas_context_top_k=max(0, int(args.ragas_context_top_k)),
        ragas_context_max_chars=max(0, int(args.ragas_context_max_chars)),
        enable_ragas=not bool(args.no_ragas),
    )

    print("\n=== Evaluation Summary ===")
    print(f"dataset: {summary['dataset_path']}")
    print(f"queries: {summary['query_count']}")
    print(f"average latency: {summary['average_latency_ms']:.2f} ms")
    print(f"p50 latency: {summary['p50_latency_ms']:.2f} ms")
    print(f"p95 latency: {summary['p95_latency_ms']:.2f} ms")
    print(f"non-empty context rate: {summary['non_empty_context_rate']:.4f}")
    print(f"run_status_counts: {summary['run_status_counts']}")
    print(f"runtime_v2_enabled: {summary['runtime_v2_enabled']}")
    print(f"fast_executor: {summary['fast_executor']}")
    print(f"per_query_timeout_seconds: {summary['per_query_timeout_seconds']}")
    print(f"ragas_enabled: {summary['ragas_enabled']}")
    print(f"ragas_timeout_seconds: {summary['ragas_timeout_seconds']}")
    print(f"ragas_context_top_k: {summary['ragas_context_top_k']}")
    print(f"ragas_context_max_chars: {summary['ragas_context_max_chars']}")
    print(f"ragas_status: {summary['ragas_status']}")
    if summary.get("ragas_error"):
        print(f"ragas_error: {summary['ragas_error']}")
    print(f"memory_sufficient_rate: {summary['memory_sufficient_rate']:.4f}")
    print(f"phase2_trigger_rate: {summary['phase2_trigger_rate']:.4f}")
    print(f"repair_trigger_rate: {summary['repair_trigger_rate']:.4f}")
    print(f"strict_fail_breakdown: {summary['strict_fail_breakdown']}")

    print("\nCustom retrieval metrics:")
    for key, value in summary["custom_metrics"].items():
        print(f"  {key}: {value:.4f}")

    print("\nRAGAS metrics:")
    for key, value in summary["ragas"].items():
        print(f"  {key}: {value:.4f}")

    print("\nBEIR precision metrics:")
    for key, value in summary["beir_precision"].items():
        print(f"  {key}: {value:.4f}")

    print(f"\noutputs: {Path(args.output_dir) / args.run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
