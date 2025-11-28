#!/usr/bin/env bash
# One-shot setup and demo run for the character KG.
# - creates/uses .venv
# - installs deps (plus fpdf2 to make a PDF from the sample text)
# - runs unit tests
# - ingests the sample PDF into Postgres and prints nodes/edges

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PDF_SOURCE_DEFAULT="$ROOT/test_files/demo_story.pdf"
TXT_SOURCE="$ROOT/test_files/demo_story.txt"
CHARACTER="${CHARACTER:-Aurelia Maren}"
WORK_NAME="${WORK_NAME:-Demo Story}"

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL is not set. Please export it (e.g., postgresql+psycopg2://user:pass@localhost:5432/kg) and re-run." >&2
  exit 1
fi

echo "Using DATABASE_URL=$DATABASE_URL"
echo "Character: $CHARACTER"

if [[ ! -d "$VENV" ]]; then
  echo "Creating virtualenv at $VENV"
  python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r "$ROOT/requirements.txt" fpdf2

PDF_PATH="${PDF_PATH:-$PDF_SOURCE_DEFAULT}"
if [[ ! -f "$PDF_PATH" ]]; then
  echo "Generating PDF from $TXT_SOURCE -> $PDF_PATH"
  python - <<'PY'
from pathlib import Path
from fpdf import FPDF

root = Path(__file__).resolve().parent.parent
txt = root / "test_files" / "demo_story.txt"
pdf_path = root / "test_files" / "demo_story.pdf"
content = txt.read_text(encoding="utf-8")

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

echo "Running tests..."
python -m unittest discover -v

echo "Running PDF demo..."
python "$ROOT/scripts/pdf_demo.py" --pdf "$PDF_PATH" --character "$CHARACTER" --work-name "$WORK_NAME"

echo "Demo complete."
