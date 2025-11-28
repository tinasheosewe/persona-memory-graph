# PDF Demo Script

`pdf_demo.py` ingests a PDF, builds the graph, and prints all nodes/edges.

## Prereqs
- Python venv with dependencies installed (`pip install -r requirements.txt`).
- Postgres reachable via `DATABASE_URL` (e.g., `postgresql+psycopg2://user:pass@localhost:5432/kg`).
- A PDF file to ingest (place under `test_files/` or anywhere).

## Usage
From repo root:
```bash
python scripts/pdf_demo.py --pdf test_files/sample.pdf --character "Marcus"
```
Options:
- `--pdf`: path to the PDF (required).
- `--character`: optional focus character to seed extraction.
- `--work-name`: optional override for the Work node name (defaults to PDF filename stem).
- `--database-url`: optional override; otherwise uses `DATABASE_URL` env var.

## What it does
1. Extracts text from the PDF with `pypdf`.
2. Runs `GraphBuilder` with `MockExtractor` (swap in `LLMExtractor` for real extraction).
3. Stores nodes/edges/segments in Postgres.
4. Prints all nodes and edges with their types, aliases, and source IDs.

## Lightweight demo (no database)
Run:
```bash
python scripts/run_demo_light.py --text test_files/demo_story.txt --character "Aurelia Maren"
# or with a PDF:
python scripts/run_demo_light.py --pdf test_files/demo_story.pdf --character "Aurelia Maren"
```
This uses `MockExtractor`, segments the text, and prints segments + extracted nodes/edges without touching a database. Install `pypdf` if you want PDF input.

## Switching to LLM extraction
Edit `scripts/pdf_demo.py`:
```python
from openai import OpenAI
from kg.pipeline import LLMExtractor
client = OpenAI()
extractor = LLMExtractor(client=client, model="gpt-4o-mini", relations=[...])
```
Then re-run the script with your `OPENAI_API_KEY` set.
