# Deep-Learner

Deep-Learner is an Agentic RAG system for evidence-grounded QA over a constrained document corpus.
It is designed to answer complex, multi-step questions with retrieval, reranking, verification, and
repair/degrade controls.

## Core Capabilities

- Agentic workflow with explicit runtime stages (`MEMORY -> PHASE1 -> PHASE2 -> REPAIR/DEGRADE -> FINALIZE`)
- Hybrid retrieval (vector + keyword) with RRF fusion
- Multi-query routing for decomposition-style retrieval
- Rerank calibration with route-aware features
- Strict verification with deterministic checks to reduce unsupported claims
- Citation-first response generation with context-window control
- Short-term memory (STM) and long-term memory (LTM) integration

## Architecture (High Level)

- `core/`: executor, state machine, node registry, contracts
- `nodes/`: runtime nodes (rewrite, retrieve, rerank, compose, verify, degrade, memory)
- `llm/`: model routing and prompt contracts
- `tools/retrieve_tool/`: vector/keyword retrieval and rerank adapters
- `ingestion/`: parsing, chunking, and dual-write indexing
- `session/` + `memory/`: session context, STM/LTM storage and recall
- `ui_streamlit.py`: interactive UI

## Tech Stack

- Python 3.10+
- LangChain model clients
- Milvus (vector storage)
- Elasticsearch (keyword/full-text retrieval)
- Streamlit (UI)

## Quick Start

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Configure environment

Create `.env` in the repository root. You can start from `.env.example`.

### 3. Run the UI

```bash
streamlit run ui_streamlit.py
```

### 4. Optional CLI mode

```bash
python -m scripts.main
```

## Notes

- This project is configured for a closed filing corpus in the current setup.
- Model routing and retrieval parameters are controlled through environment variables.
- For reproducibility, keep the same `.env` and index state across runs.

## Author

group9

