# DB Profiles (Pinned for Reuse)

These two database profiles are now the only test corpora to use.
Do not run ingestion again unless explicitly requested.

## 1) Short Docs Profile

- `ES_INDEX_NAME=deep_learner_knowledge_short_docs_v1`
- `MILVUS_COLLECTION_NAME=deep_learner_vectors_short_docs_v1`
- `MILVUS_LTM_COLLECTION=user_long_term_memory_short_docs_v1`
- Source docs path used for ingestion:
  - `evaluation_suite/hf_data/squad_100/docs`
- Current counts:
  - ES docs: `100`
  - Milvus vectors: `100`
  - Milvus LTM: `0`

Use this profile:

```bash
export ES_INDEX_NAME=deep_learner_knowledge_short_docs_v1
export MILVUS_COLLECTION_NAME=deep_learner_vectors_short_docs_v1
export MILVUS_LTM_COLLECTION=user_long_term_memory_short_docs_v1
```

## 2) Long Docs Profile

- `ES_INDEX_NAME=deep_learner_knowledge_long_docs_v1`
- `MILVUS_COLLECTION_NAME=deep_learner_vectors_long_docs_v1`
- `MILVUS_LTM_COLLECTION=user_long_term_memory_long_docs_v1`
- Source docs path used for ingestion:
  - `evaluation_suite/hf_data/triviaqa_long_100/docs`
- Current counts:
  - ES docs: `5421`
  - Milvus vectors: `5457`
  - Milvus LTM: `0`

Use this profile:

```bash
export ES_INDEX_NAME=deep_learner_knowledge_long_docs_v1
export MILVUS_COLLECTION_NAME=deep_learner_vectors_long_docs_v1
export MILVUS_LTM_COLLECTION=user_long_term_memory_long_docs_v1
```

## Validation Snapshot

After cleanup + ingestion, only these DB artifacts exist:

- ES indices:
  - `deep_learner_knowledge_short_docs_v1`
  - `deep_learner_knowledge_long_docs_v1`
- Milvus collections:
  - `deep_learner_vectors_short_docs_v1`
  - `deep_learner_vectors_long_docs_v1`
  - `user_long_term_memory_short_docs_v1`
  - `user_long_term_memory_long_docs_v1`
