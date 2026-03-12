# Deep-Learner

Deep-Learner is an Agentic RAG system for evidence-grounded QA over a constrained filing corpus.

## What Is Included

- Runtime pipeline with staged control flow: `MEMORY -> PHASE1 -> PHASE2 -> REPAIR/DEGRADE -> FINALIZE`
- Hybrid retrieval: vector + keyword + RRF + rerank
- Strict verification and degrade fallback
- Streamlit frontend
- Curated corpus committed in repo:
  - `data/10k/Amazon 10K 2024.pdf`
  - `data/10k/Alphabet 10K 2024.pdf`
  - `data/10k/MSFT 10-K.pdf`

## Prerequisites

- Python 3.10+
- Docker + Docker Compose
- At least one LLM API key (OpenAI / Anthropic / Gemini / Ollama local)

## Quick Start (Recommended)

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
```

Then fill API keys in `.env` (for example `OPENAI_API_KEY=...`).

Important:
- Do not commit `.env` or real API keys to GitHub.
- Keep `.env.example` as the shareable template.

### 3. One-click rebuild of local databases (fast + pdfplumber)

This starts Docker services and rebuilds Milvus + Elasticsearch indexes using:
- `UNSTRUCTURED_STRATEGY=fast`
- `PDF_EXTRACT_TABLES_WITH_PDFPLUMBER=true`
- `UNSTRUCTURED_INFER_TABLE_STRUCTURE=false`

```bash
./scripts/one_click_rebuild_10k.sh
```

### 4. Run frontend

```bash
streamlit run ui_streamlit.py
```

## Manual Commands (If Needed)

Start infrastructure:

```bash
docker compose -f docker/docker-compose.yml up -d
```

Rebuild indexes only:

```bash
UNSTRUCTURED_STRATEGY=fast \
UNSTRUCTURED_INFER_TABLE_STRUCTURE=false \
PDF_EXTRACT_TABLES_WITH_PDFPLUMBER=true \
DOCS_PATH=data/10k \
INGEST_RESET_TARGET=true \
python -m scripts.rebuild_databases
```

Reset databases:

```bash
python -m scripts.reset_databases
```

## Health Checks

- Elasticsearch: `http://localhost:9200`
- Milvus port: `localhost:19530`

## Reproducibility Notes

- Vector/keyword databases are local runtime state and are not shared automatically across users.
- With this repo, users can reproduce the same index locally from `data/10k` via the one-click script.
- Keep `.env` settings and rebuild strategy consistent for comparable behavior.

## Author

group9
