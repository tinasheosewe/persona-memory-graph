# Demo Scripts

The Python scripts import the `kg` package, so run them from the repository root with the root on `PYTHONPATH` (as in the examples below). The shell wrappers (`run_demo.sh`, `run_demo_pdf.sh`, `run_api.sh`, `run_ingest_neo4j.sh`, `dump_graph.sh`) set it themselves.

## PDF Demo Script

`pdf_demo.py` ingests a PDF, builds the graph, and prints all nodes/edges.

### Prereqs
- Python venv with dependencies installed (`pip install -r requirements.txt`).
- Postgres reachable via `DATABASE_URL` (e.g., `postgresql+psycopg2://user:pass@localhost:5432/kg`).
- A PDF file to ingest (`test_files/demo_story.pdf` is the sample story as a PDF, or use your own).

### Usage
From repo root:
```bash
PYTHONPATH=. python scripts/pdf_demo.py --pdf test_files/demo_story.pdf --character "Aurelia Maren"
```
Options:
- `--pdf`: path to the PDF (required).
- `--character`: optional focus character to seed extraction.
- `--work-name`: optional override for the Work node name (defaults to PDF filename stem).
- `--database-url`: optional override; otherwise uses `DATABASE_URL` env var.

### What it does
1. Extracts text from the PDF with `pypdf`.
2. Runs `GraphBuilder` with `MockExtractor` (swap in `LLMExtractor` for real extraction).
3. Stores nodes/edges/segments in Postgres.
4. Prints all nodes and edges with their types, aliases, and source IDs.

## Lightweight demo (no database)
Run:
```bash
PYTHONPATH=. python scripts/run_demo_light.py --text test_files/demo_story.txt --character "Aurelia Maren"
# or with a PDF:
PYTHONPATH=. python scripts/run_demo_light.py --pdf test_files/demo_story.pdf --character "Aurelia Maren"
```
This uses `MockExtractor` by default, segments the text, and prints segments + extracted nodes/edges without touching a database.

### Using the LLM extractor (OpenAI)
Set your key and optionally a model, then pass `--use-llm`:
```bash
export OPENAI_API_KEY=sk-...
export DEMO_LLM_MODEL=gpt-4o-mini   # optional
PYTHONPATH=. python scripts/run_demo_light.py --text test_files/demo_story.txt --character "Aurelia Maren" --use-llm
```
The relation vocabulary can be overridden via `--relations REL1 REL2 ...` (names from `kg/types.py`).

### One-shot script
`scripts/run_demo.sh` creates `.venv`, installs deps, runs the tests, and runs the lightweight demo.
Defaults to text input and `MockExtractor`.
```bash
./scripts/run_demo.sh
```
To use the LLM demo:
```bash
OPENAI_API_KEY=sk-... USE_LLM=1 ./scripts/run_demo.sh
```
To switch to the PDF path (`scripts/run_demo_pdf.sh` is a shortcut for this):
```bash
USE_PDF=1 ./scripts/run_demo.sh
```

## Neo4j helpers
- `ingest_to_neo4j.py` / `run_ingest_neo4j.sh` ingest a text or PDF straight into Neo4j (needs `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`) and print the extracted nodes and edges.
- `query_neo4j_demo.py` fetches the graph for a `character_key` from Neo4j and asks the LLM to answer a prompt from it (needs `OPENAI_API_KEY`).
- `dump_graph.sh` writes the Postgres graph to `graph_dump.json`.

## Switching to LLM extraction
Edit `scripts/pdf_demo.py`:
```python
from openai import OpenAI
from kg.pipeline import LLMExtractor
client = OpenAI()
extractor = LLMExtractor(client=client, model="gpt-4o-mini")
```
Then re-run the script with your `OPENAI_API_KEY` set.
