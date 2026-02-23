#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return value or "doc"


def _doc_filename(title: str, context: str) -> str:
    digest = hashlib.sha1(context.encode("utf-8")).hexdigest()[:10]
    return f"{_slug(title)}_{digest}.txt"


def _extract_answer(row: dict[str, Any]) -> str:
    answers = row.get("answers", {})
    if isinstance(answers, dict):
        texts = answers.get("text", [])
        if isinstance(texts, list) and texts:
            return str(texts[0]).strip()
    return ""


def prepare(
    hf_dataset: str,
    split: str,
    num_queries: int,
    output_dir: Path,
    shuffle: bool,
    seed: int,
) -> dict[str, Any]:
    ds = load_dataset(hf_dataset, split=split)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    ds = ds.select(range(min(num_queries, len(ds))))

    docs_dir = output_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    testset_path = output_dir / "testset.jsonl"
    manifest_path = output_dir / "manifest.json"

    context_to_doc: dict[str, str] = {}
    manifest_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for i, row in enumerate(ds, start=1):
        question = str(row.get("question", "")).strip()
        context = str(row.get("context", "")).strip()
        title = str(row.get("title", "squad")).strip()
        answer = _extract_answer(row)

        if not question or not context:
            continue

        doc_name = context_to_doc.get(context)
        if not doc_name:
            doc_name = _doc_filename(title=title, context=context)
            context_to_doc[context] = doc_name
            content = f"# {title}\n\n{context}\n"
            (docs_dir / doc_name).write_text(content, encoding="utf-8")

        qid = str(row.get("id", f"squad_{i:04d}")).strip() or f"squad_{i:04d}"

        test_rows.append(
            {
                "id": qid,
                "question": question,
                "ground_truth_answer": answer or context[:160],
                "ground_truth_docs": [doc_name],
            }
        )

        manifest_rows.append(
            {
                "id": qid,
                "title": title,
                "doc_name": doc_name,
                "question_preview": question[:120],
            }
        )

    with testset_path.open("w", encoding="utf-8") as f:
        for row in test_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "hf_dataset": hf_dataset,
        "split": split,
        "requested_queries": num_queries,
        "final_queries": len(test_rows),
        "unique_docs": len(context_to_doc),
        "docs_dir": str(docs_dir),
        "testset_path": str(testset_path),
        "rows": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare docs + testset (question/ground_truth_answer/ground_truth_docs) from Hugging Face SQuAD."
    )
    parser.add_argument("--hf-dataset", default="rajpurkar/squad", help="HF dataset name")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--num-queries", type=int, default=20, help="Number of queries to build")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffle before selecting queries")
    parser.add_argument(
        "--output-dir",
        default="evaluation_suite/hf_data/squad_20",
        help="Output directory for docs/testset",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = prepare(
        hf_dataset=args.hf_dataset,
        split=args.split,
        num_queries=args.num_queries,
        output_dir=Path(args.output_dir),
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )
    print(f"Prepared from HF dataset: {manifest['hf_dataset']} ({manifest['split']})")
    print(f"queries: {manifest['final_queries']}, unique_docs: {manifest['unique_docs']}")
    print(f"docs_dir: {manifest['docs_dir']}")
    print(f"testset: {manifest['testset_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
