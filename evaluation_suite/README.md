# Evaluation Suite

This directory provides a complete evaluation pipeline covering the combinations you require:

- `RAGAS`: Generate quality indicators
- `latency`: end-to-end time-consuming statistics
- `retrieval recall@k`: Custom recall evaluation
- `HuggingFace Dataset`: test set management
- `BEIR Benchmark`: retrieval accuracy/sorting index (including `P@k`)

## 1) Test set format (required)

Each sample contains at least:

```json
{
  "question": "...",
  "ground_truth_answer": "...",
  "ground_truth_docs": ["doc_a.md", "doc_b.pdf"]
}
```

Optional fields:

- `id`: sample ID (if left blank, it will be automatically generated)

Default sample data:

- `evaluation_suite/datasets/testset_20.jsonl`

## 2) Execute full evaluation with one click

```bash
cd /Users/justin/Desktop/Deep-Learner
PYTHONPATH=. .venv/bin/python evaluation_suite/run_eval.py \
  --dataset evaluation_suite/datasets/testset_20.jsonl \
  --output-dir evaluation_suite/outputs \
  --k-values 1,3,5
```

illustrate:

- LTMs are isolated by default (to prevent reviews from contaminating each other).
- To run according to online behavior (enable LTM reading and writing), add `--no-isolate-ltm`.
- Runtime V2 is enabled by default (Memory-First + Evidence-Driven state machine).
- Disable each question sub-process timeout by default (avoiding `spawn` overhead and making batch assessment faster).
- `fast_executor` only takes effect on legacy runtime.
- To run the complete link (including verify/repair) and enable hard timeout, add:

```bash
PYTHONPATH=. .venv/bin/python evaluation_suite/run_eval.py \
  --dataset evaluation_suite/datasets/testset_20.jsonl \
  --output-dir evaluation_suite/outputs \
  --no-fast-executor \
  --per-query-timeout-seconds 120
```

RAGAS stability parameters (anti-jamming):

- `--ragas-timeout-seconds 240`: The RAGAS phase will automatically downgrade to `NaN` indicator after timeout, without blocking the main evaluation output.
- `--ragas-context-top-k 8`: Only pass the first K contexts to RAGAS.
- `--ragas-context-max-chars 1200`: Truncate length per context.
- `--no-ragas`: Turn off RAGAS completely, leaving only link/retrieval metrics.

## 3) Output results

Each run will generate a directory: `evaluation_suite/outputs/<run_name>/`

- `summary.json`: total metrics (RAGAS + latency + recall@k + BEIR)
- `records.jsonl`: sample results one by one (answer, retrieved docs, latency, various scores)
- `records.csv`: a flat table that can be directly analyzed
- `stats_table.md`: Statistics table (Markdown)

Runtime V2 additional summary metrics:

- `memory_sufficient_rate`
- `phase2_trigger_rate`
- `repair_trigger_rate`
- `strict_fail_breakdown`

## 4) Data set management (HuggingFace)

### Verification data set

```bash
PYTHONPATH=. .venv/bin/python evaluation_suite/dataset_manager.py validate \
  --dataset evaluation_suite/datasets/testset_20.jsonl
```

### Convert to HuggingFace disk dataset

```bash
PYTHONPATH=. .venv/bin/python evaluation_suite/dataset_manager.py to-hf \
  --dataset evaluation_suite/datasets/testset_20.jsonl \
  --out-dir evaluation_suite/datasets/hf_testset
```

### Export standardized version

```bash
PYTHONPATH=. .venv/bin/python evaluation_suite/dataset_manager.py export-normalized \
  --dataset evaluation_suite/datasets/testset_20.jsonl \
  --output evaluation_suite/datasets/testset_20.normalized.jsonl
```

## 5) Current implementation indicators

### RAGAS

- `answer_relevancy`
- `context_precision`
- `context_recall`
- `answer_correctness`

### Custom indicators

- `average/p50/p95 latency`
- `recall@k`
- `precision@k`

### BEIR indicator

- `NDCG@k`
- `MAP@k`
- `Recall@k`
- `P@k` (retrieval accuracy)
