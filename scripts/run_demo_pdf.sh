#!/usr/bin/env bash
# Convenience wrapper to run the lightweight demo against demo_story.pdf.
# Reuses scripts/run_demo.sh but forces USE_PDF=1 so the PDF pipeline path is executed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PDF_PATH_DEFAULT="$ROOT/test_files/demo_story.pdf"

export USE_PDF=1
export PDF_PATH="${PDF_PATH:-$PDF_PATH_DEFAULT}"
export CHARACTER="${CHARACTER:-Warren Buffett}"
export WORK_NAME="${WORK_NAME:-Demo Story}"

exec "$SCRIPT_DIR/run_demo.sh"
