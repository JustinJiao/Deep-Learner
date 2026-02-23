# Evaluation Suite

这个目录提供一套完整评测流水线，覆盖你要求的组合：

- `RAGAS`：生成质量指标
- `latency`：端到端耗时统计
- `retrieval recall@k`：自定义召回评估
- `HuggingFace Dataset`：测试集管理
- `BEIR Benchmark`：检索精度/排序指标（含 `P@k`）

## 1) 测试集格式（必需）

每条样本至少包含：

```json
{
  "question": "...",
  "ground_truth_answer": "...",
  "ground_truth_docs": ["doc_a.md", "doc_b.pdf"]
}
```

可选字段：

- `id`: 样本 ID（不填会自动生成）

默认示例数据：

- `evaluation_suite/datasets/testset_20.jsonl`

## 2) 一键执行全评测

```bash
cd /Users/justin/Desktop/Deep-Learner
PYTHONPATH=. .venv/bin/python evaluation_suite/run_eval.py \
  --dataset evaluation_suite/datasets/testset_20.jsonl \
  --output-dir evaluation_suite/outputs \
  --k-values 1,3,5
```

说明：

- 默认会隔离 LTM（防止评测互相污染）。
- 若要按线上行为运行（开启 LTM 读写），加 `--no-isolate-ltm`。

## 3) 输出结果

每次运行会生成目录：`evaluation_suite/outputs/<run_name>/`

- `summary.json`：总指标（RAGAS + latency + recall@k + BEIR）
- `records.jsonl`：逐条样本结果（answer、retrieved docs、latency、各项分数）
- `records.csv`：可直接分析的平面表
- `stats_table.md`：统计表（Markdown）

## 4) 数据集管理（HuggingFace）

### 校验数据集

```bash
PYTHONPATH=. .venv/bin/python evaluation_suite/dataset_manager.py validate \
  --dataset evaluation_suite/datasets/testset_20.jsonl
```

### 转为 HuggingFace disk dataset

```bash
PYTHONPATH=. .venv/bin/python evaluation_suite/dataset_manager.py to-hf \
  --dataset evaluation_suite/datasets/testset_20.jsonl \
  --out-dir evaluation_suite/datasets/hf_testset
```

### 导出标准化版本

```bash
PYTHONPATH=. .venv/bin/python evaluation_suite/dataset_manager.py export-normalized \
  --dataset evaluation_suite/datasets/testset_20.jsonl \
  --output evaluation_suite/datasets/testset_20.normalized.jsonl
```

## 5) 当前实现指标

### RAGAS

- `answer_relevancy`
- `context_precision`
- `context_recall`
- `answer_correctness`

### 自定义指标

- `average/p50/p95 latency`
- `recall@k`
- `precision@k`

### BEIR 指标

- `NDCG@k`
- `MAP@k`
- `Recall@k`
- `P@k`（检索精度）
