#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

REQUIRED_FIELDS = ("question", "ground_truth_answer", "ground_truth_docs")


def _normalize_doc_id(raw: Any) -> str:
    return Path(str(raw or "").strip()).name.lower()


def _parse_docs(raw: Any) -> list[str]:
    if isinstance(raw, list):
        docs = [_normalize_doc_id(x) for x in raw if str(x).strip()]
    elif isinstance(raw, str):
        docs = [_normalize_doc_id(x) for x in raw.split(",") if x.strip()]
    else:
        docs = []
    return sorted(set(docs))


def load_rows(dataset_path: Path) -> list[dict[str, Any]]:
    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ds, start=1):
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            raise ValueError(f"row {idx} missing fields: {missing}")

        question = str(row.get("question", "")).strip()
        gt_answer = str(row.get("ground_truth_answer", "")).strip()
        gt_docs = _parse_docs(row.get("ground_truth_docs"))
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
    return rows


def cmd_validate(args: argparse.Namespace) -> int:
    rows = load_rows(Path(args.dataset))
    doc_set = sorted({doc for row in rows for doc in row["ground_truth_docs"]})
    print("dataset valid")
    print(f"rows: {len(rows)}")
    print(f"unique docs: {len(doc_set)}")
    print(f"docs: {doc_set}")
    return 0


def cmd_to_hf(args: argparse.Namespace) -> int:
    rows = load_rows(Path(args.dataset))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(out_dir))
    print(f"saved huggingface dataset: {out_dir}")
    print(f"rows: {len(rows)}")
    return 0


def cmd_export_normalized(args: argparse.Namespace) -> int:
    rows = load_rows(Path(args.dataset))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"exported normalized dataset: {out_path}")
    print(f"rows: {len(rows)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage evaluation dataset with HuggingFace Dataset API.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_validate = sub.add_parser("validate", help="Validate dataset schema and values")
    p_validate.add_argument("--dataset", default="evaluation_suite/datasets/testset_20.jsonl")
    p_validate.set_defaults(func=cmd_validate)

    p_to_hf = sub.add_parser("to-hf", help="Convert dataset to HuggingFace disk format")
    p_to_hf.add_argument("--dataset", default="evaluation_suite/datasets/testset_20.jsonl")
    p_to_hf.add_argument("--out-dir", default="evaluation_suite/datasets/hf_testset")
    p_to_hf.set_defaults(func=cmd_to_hf)

    p_export = sub.add_parser("export-normalized", help="Normalize and export dataset")
    p_export.add_argument("--dataset", default="evaluation_suite/datasets/testset_20.jsonl")
    p_export.add_argument("--output", default="evaluation_suite/datasets/testset_20.normalized.jsonl")
    p_export.set_defaults(func=cmd_export_normalized)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
