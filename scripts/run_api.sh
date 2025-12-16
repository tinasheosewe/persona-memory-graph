#!/usr/bin/env bash
# Simple runner to launch the FastAPI service with uvicorn.
# Loads .env if present, and uses .venv if available.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
VENV="$ROOT/.venv"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

if [[ -d "$VENV" ]]; then
  source "$VENV/bin/activate"
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8001}"
