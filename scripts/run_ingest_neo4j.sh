#!/usr/bin/env bash
# Convenience runner: ingest a text/PDF into Neo4j and print nodes/edges.
# Defaults to test_files/demo_story.txt and character_key=default.

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

TEXT_PATH="${1:-$ROOT/test_files/demo_story.txt}"
CHARACTER="${CHARACTER:-Aurelia}"
CHARACTER_KEY="${CHARACTER_KEY:-default}"

python "$ROOT/scripts/ingest_to_neo4j.py" --text "$TEXT_PATH" --character "$CHARACTER" --character-key "$CHARACTER_KEY"
