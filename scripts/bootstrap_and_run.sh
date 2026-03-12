#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f ".env" ]]; then
  cp .env.example .env
  echo "Created .env from .env.example."
  echo "Please set at least one API key in .env (for example OPENAI_API_KEY), then re-run:"
  echo "  ./scripts/bootstrap_and_run.sh"
  exit 1
fi

if ! grep -Eq '^(OPENAI_API_KEY|ANTHROPIC_API_KEY|GEMINI_API_KEY)=[^[:space:]]+' .env; then
  echo "No API key found in .env."
  echo "Please set at least one of OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY, then re-run."
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -e .

./scripts/one_click_rebuild_10k.sh

echo ""
echo "Starting frontend..."
streamlit run ui_streamlit.py

