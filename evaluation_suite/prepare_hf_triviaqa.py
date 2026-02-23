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


def _doc_filename(filename: str, context: str) -> str:
    stem = Path(filename).stem or "doc"
    digest = hashlib.sha1(context.encode("utf-8")).hexdigest()[:10]
    return f"{_slug(stem)}_{digest}.txt"


def _extract_answer(row: dict[str, Any]) -> str:
    answer = row.get("answer", {})
    if isinstance(answer, dict):
        value = str(answer.get("value", "") or "").strip()
        if value:
            return value
        aliases = answer.get("aliases", [])
        if isinstance(aliases, list) and aliases:
            return str(aliases[0]).strip()
    return ""


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def prepare(
    hf_dataset: str,
    hf_config: str,
    split: str,
    num_queries: int,
    output_dir: Path,
    shuffle: bool,
    seed: int,
    min_doc_chars: int,
    require_long_docs: bool,
) -> dict[str, Any]:
    ds = load_dataset(hf_dataset, hf_config, split=split)
    if shuffle:
        ds = ds.shuffle(seed=seed)

    docs_dir = output_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    testset_path = output_dir / "testset.jsonl"
    manifest_path = output_dir / "manifest.json"

    content_hash_to_doc_name: dict[str, str] = {}
    test_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    skipped_no_docs = 0
    skipped_no_answer = 0
    skipped_not_long = 0

    for i, row in enumerate(ds, start=1):
        if len(test_rows) >= num_queries:
            break

        question = str(row.get("question", "")).strip()
        question_id = str(row.get("question_id", f"triviaqa_{i:05d}")).strip() or f"triviaqa_{i:05d}"
        answer = _extract_answer(row)

        entity_pages = row.get("entity_pages", {}) or {}
        filenames = entity_pages.get("filename", []) or []
        titles = entity_pages.get("title", []) or []
        contexts = entity_pages.get("wiki_context", []) or []

        if not question:
            continue
        if not answer:
            skipped_no_answer += 1
            continue

        usable_items: list[tuple[str, str, str]] = []
        long_items: list[tuple[str, str, str]] = []

        n = min(len(filenames), len(titles), len(contexts))
        for j in range(n):
            filename = str(filenames[j] or f"doc_{j}.txt").strip()
            title = str(titles[j] or Path(filename).stem).strip()
            context = str(contexts[j] or "").strip()
            if not context:
                continue
            item = (filename, title, context)
            usable_items.append(item)
            if len(context) >= min_doc_chars:
                long_items.append(item)

        if not usable_items:
            skipped_no_docs += 1
            continue

        selected_items = long_items if long_items else usable_items
        if require_long_docs and not long_items:
            skipped_not_long += 1
            continue

        ground_truth_docs: list[str] = []
        selected_doc_char_lens: list[int] = []
        for filename, title, context in selected_items:
            content_hash = hashlib.sha1(context.encode("utf-8")).hexdigest()
            doc_name = content_hash_to_doc_name.get(content_hash)
            if not doc_name:
                doc_name = _doc_filename(filename=filename, context=context)
                content_hash_to_doc_name[content_hash] = doc_name
                content = f"# {title}\n\n{context}\n"
                (docs_dir / doc_name).write_text(content, encoding="utf-8")

            ground_truth_docs.append(doc_name)
            selected_doc_char_lens.append(len(context))

        ground_truth_docs = _dedup_keep_order(ground_truth_docs)
        if not ground_truth_docs:
            skipped_no_docs += 1
            continue

        test_rows.append(
            {
                "id": question_id,
                "question": question,
                "ground_truth_answer": answer,
                "ground_truth_docs": ground_truth_docs,
            }
        )

        manifest_rows.append(
            {
                "id": question_id,
                "question_preview": question[:120],
                "answer_preview": answer[:120],
                "ground_truth_docs_count": len(ground_truth_docs),
                "ground_truth_docs": ground_truth_docs,
                "selected_doc_char_lens": selected_doc_char_lens,
            }
        )

    if len(test_rows) < num_queries:
        raise ValueError(
            "Not enough rows collected. "
            f"requested={num_queries}, built={len(test_rows)}, "
            f"skipped_no_docs={skipped_no_docs}, skipped_no_answer={skipped_no_answer}, "
            f"skipped_not_long={skipped_not_long}, min_doc_chars={min_doc_chars}"
        )

    with testset_path.open("w", encoding="utf-8") as f:
        for row in test_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    doc_char_lens: list[int] = []
    for m in manifest_rows:
        doc_char_lens.extend(m.get("selected_doc_char_lens", []))

    manifest = {
        "hf_dataset": hf_dataset,
        "hf_config": hf_config,
        "split": split,
        "requested_queries": num_queries,
        "final_queries": len(test_rows),
        "unique_docs": len(content_hash_to_doc_name),
        "min_doc_chars": min_doc_chars,
        "require_long_docs": require_long_docs,
        "docs_dir": str(docs_dir),
        "testset_path": str(testset_path),
        "avg_selected_doc_chars": (sum(doc_char_lens) / len(doc_char_lens)) if doc_char_lens else 0.0,
        "max_selected_doc_chars": max(doc_char_lens) if doc_char_lens else 0,
        "min_selected_doc_chars": min(doc_char_lens) if doc_char_lens else 0,
        "rows": manifest_rows,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare long-document testset from Hugging Face TriviaQA (rc.wikipedia).",
    )
    parser.add_argument("--hf-dataset", default="trivia_qa", help="HF dataset name")
    parser.add_argument("--hf-config", default="rc.wikipedia", help="HF dataset config")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--num-queries", type=int, default=20, help="Number of queries to build")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffle before selecting queries")
    parser.add_argument(
        "--min-doc-chars",
        type=int,
        default=15000,
        help="Only keep docs with at least this many characters as long docs",
    )
    parser.add_argument(
        "--allow-non-long-fallback",
        action="store_true",
        help="If set, rows without long docs can still be used by falling back to available docs",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_suite/hf_data/triviaqa_long_20",
        help="Output directory for docs/testset",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = prepare(
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        split=args.split,
        num_queries=args.num_queries,
        output_dir=Path(args.output_dir),
        shuffle=not args.no_shuffle,
        seed=args.seed,
        min_doc_chars=args.min_doc_chars,
        require_long_docs=not args.allow_non_long_fallback,
    )
    print(
        f"Prepared from HF dataset: {manifest['hf_dataset']} "
        f"({manifest['hf_config']}/{manifest['split']})"
    )
    print(f"queries: {manifest['final_queries']}, unique_docs: {manifest['unique_docs']}")
    print(
        "selected doc chars: "
        f"min={manifest['min_selected_doc_chars']}, "
        f"avg={manifest['avg_selected_doc_chars']:.2f}, "
        f"max={manifest['max_selected_doc_chars']}"
    )
    print(f"docs_dir: {manifest['docs_dir']}")
    print(f"testset: {manifest['testset_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
