#!/usr/bin/env bash
# Dump the current Postgres-backed graph to a JSON file.
# Usage:
#   scripts/dump_graph.sh [output_path]
# Requires: DATABASE_URL set; will read .env if present.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
OUT="${1:-$ROOT/graph_dump.json}"
VENV="$ROOT/.venv"

# Load environment from .env if present
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT/.env"
  set +a
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL is required to dump the graph." >&2
  exit 1
fi

if [[ -d "$VENV" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OUT_PATH="$OUT"

python - <<'PY'
import json
import os
from datetime import datetime
from pathlib import Path

from kg.schema import KGBasicNode, KGEdge, SourceSegment, create_engine_and_session

database_url = os.environ["DATABASE_URL"]
engine, SessionLocal = create_engine_and_session(database_url)
session = SessionLocal()

def serialize_node(node: KGBasicNode) -> dict:
    return {
        "id": str(node.id),
        "character_key": node.character_key,
        "type": node.type,
        "name": node.name,
        "alias_names": node.alias_names or [],
        "summary": node.summary,
        "meta": node.meta or {},
        "source_ids": node.source_ids or [],
        "created_at": node.created_at.isoformat() if node.created_at else None,
        "updated_at": node.updated_at.isoformat() if node.updated_at else None,
    }

def serialize_edge(edge: KGEdge) -> dict:
    return {
        "id": edge.id,
        "character_key": edge.character_key,
        "from_id": str(edge.from_id) if edge.from_id else None,
        "to_id": str(edge.to_id) if edge.to_id else None,
        "relation": edge.relation,
        "weight": edge.weight,
        "description": edge.description,
        "meta": edge.meta or {},
        "source_ids": edge.source_ids or [],
        "created_at": edge.created_at.isoformat() if edge.created_at else None,
    }

def serialize_segment(seg: SourceSegment) -> dict:
    return {
        "id": seg.id,
        "work_id": str(seg.work_id) if seg.work_id else None,
        "location": seg.location,
        "content": seg.content,
        "meta": seg.meta or {},
        "created_at": seg.created_at.isoformat() if seg.created_at else None,
    }

nodes = [serialize_node(n) for n in session.query(KGBasicNode).all()]
edges = [serialize_edge(e) for e in session.query(KGEdge).all()]
segments = [serialize_segment(s) for s in session.query(SourceSegment).all()]

session.close()

dump = {
    "exported_at": datetime.utcnow().isoformat() + "Z",
    "nodes": nodes,
    "edges": edges,
    "segments": segments,
}

out_path = Path(os.environ.get("OUT_PATH", "")) if os.environ.get("OUT_PATH") else Path("graph_dump.json")
out_path.write_text(json.dumps(dump, ensure_ascii=False, indent=2))
print(f"Wrote graph dump to {out_path}")
PY
