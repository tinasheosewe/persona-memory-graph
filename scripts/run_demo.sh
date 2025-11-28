#!/usr/bin/env bash
# One-shot setup and lightweight demo (no Postgres required).
# - creates/uses .venv
# - installs deps (plus fpdf2 to make a PDF from the sample text if desired)
# - runs unit tests
# - runs the lightweight demo to print segments/nodes/edges (using text by default)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$ROOT/.venv"
TXT_SOURCE="$ROOT/test_files/demo_story.txt"
PDF_SOURCE_DEFAULT="$ROOT/test_files/demo_story.pdf"
CHARACTER="${CHARACTER:-Aurelia Maren}"
WORK_NAME="${WORK_NAME:-Demo Story}"

# Load environment from .env if present (for OPENAI_API_KEY / DEMO_LLM_MODEL)
if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

if [[ ! -d "$VENV" ]]; then
  echo "Creating virtualenv at $VENV"
  python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"
# Ensure local package is importable
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r "$ROOT/requirements.txt" fpdf2

# Prefer text demo; generate PDF only if requested
USE_PDF="${USE_PDF:-0}"
PDF_PATH="${PDF_PATH:-$PDF_SOURCE_DEFAULT}"

if [[ "$USE_PDF" == "1" ]]; then
  if [[ ! -f "$PDF_PATH" ]]; then
    echo "Generating PDF from $TXT_SOURCE -> $PDF_PATH"
    DEMO_ROOT="$ROOT" python - <<'PY'
import os
from pathlib import Path
from fpdf import FPDF

root = Path(os.environ["DEMO_ROOT"])
txt = root / "test_files" / "demo_story.txt"
pdf_path = root / "test_files" / "demo_story.pdf"
txt.parent.mkdir(parents=True, exist_ok=True)
if not txt.exists():
    raise FileNotFoundError(f"Missing text source: {txt}")
content = txt.read_text(encoding="utf-8")
# Normalize curly quotes to avoid font issues with built-in fonts
content = (
    content.replace("“", '"')
    .replace("”", '"')
    .replace("’", "'")
)
pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)
for line in content.splitlines():
    pdf.multi_cell(0, 10, line)
pdf.output(str(pdf_path))
print(f"Wrote {pdf_path}")
PY
  else
    echo "Using existing PDF at $PDF_PATH"
  fi
fi

echo "Running tests..."
python -m unittest discover -v

echo "Running lightweight demo..."
LLM_FLAG=()
USE_LLM="${USE_LLM:-${OPENAI_API_KEY:+1}}"
if [[ "$USE_LLM" == "1" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required when USE_LLM=1" >&2
    exit 1
  fi
  LLM_FLAG=(--use-llm --model "${LLM_MODEL:-gpt-4o-mini}")
fi

if [[ "$USE_PDF" == "1" ]]; then
  python "$ROOT/scripts/run_demo_light.py" --pdf "$PDF_PATH" --character "$CHARACTER" --work-name "$WORK_NAME" "${LLM_FLAG[@]}"
else
  python "$ROOT/scripts/run_demo_light.py" --text "$TXT_SOURCE" --character "$CHARACTER" --work-name "$WORK_NAME" "${LLM_FLAG[@]}"
fi

echo "Demo complete."
