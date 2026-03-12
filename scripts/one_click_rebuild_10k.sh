#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

COMPOSE_FILE="docker/docker-compose.yml"

echo "[1/4] Starting infrastructure (Milvus + Elasticsearch)..."
docker compose -f "${COMPOSE_FILE}" up -d

echo "[2/4] Waiting for Elasticsearch (localhost:9200)..."
for i in {1..90}; do
  if curl -fsS "http://localhost:9200" >/dev/null 2>&1; then
    echo "Elasticsearch is ready."
    break
  fi
  sleep 2
  if [[ "${i}" -eq 90 ]]; then
    echo "Elasticsearch did not become ready in time."
    exit 1
  fi
done

echo "[3/4] Waiting for Milvus (localhost:19530)..."
for i in {1..90}; do
  if nc -z localhost 19530 >/dev/null 2>&1; then
    echo "Milvus is ready."
    break
  fi
  sleep 2
  if [[ "${i}" -eq 90 ]]; then
    echo "Milvus did not become ready in time."
    exit 1
  fi
done

echo "[4/4] Rebuilding indexes with fast + pdfplumber strategy..."
UNSTRUCTURED_STRATEGY=fast \
UNSTRUCTURED_INFER_TABLE_STRUCTURE=false \
PDF_EXTRACT_TABLES_WITH_PDFPLUMBER=true \
DOCS_PATH=data/10k \
INGEST_RESET_TARGET=true \
python -m scripts.rebuild_databases

echo ""
echo "Done. You can now run:"
echo "streamlit run ui_streamlit.py"

