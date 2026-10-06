"""
Demo script: ingest a PDF, build the graph, and print nodes/edges.

Usage (from the repository root):
  PYTHONPATH=. python scripts/pdf_demo.py --pdf test_files/demo_story.pdf --character "Aurelia Maren"
Environment:
  - DATABASE_URL must point to Postgres (e.g., postgresql+psycopg2://user:pass@localhost:5432/kg)
Extractor:
  - Uses MockExtractor by default; swap in LLMExtractor if you want real entity/relation extraction.
"""

import argparse
import os
from pathlib import Path

from pypdf import PdfReader

from kg.pipeline import GraphBuilder, MockExtractor
from kg.schema import KGBasicNode, KGEdge, create_engine_and_session


def read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def print_nodes_and_edges(session):
    nodes = session.query(KGBasicNode).order_by(KGBasicNode.type, KGBasicNode.name).all()
    id_to_name = {n.id: n.name for n in nodes}
    print("\nNodes:")
    for n in nodes:
        print(f"- [{n.type}] {n.name} aliases={n.alias_names or []} sources={n.source_ids or []}")

    edges = session.query(KGEdge).all()
    print("\nEdges:")
    for e in edges:
        print(
            f"- {id_to_name.get(e.from_id, e.from_id)} -[{e.relation}]-> {id_to_name.get(e.to_id, e.to_id)} "
            f"src={e.source_ids or []} desc={e.description or ''}"
        )


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF and print graph contents.")
    parser.add_argument("--pdf", required=True, help="Path to PDF file (e.g., test_files/demo_story.pdf)")
    parser.add_argument("--character", help="Focus character name to seed the extractor")
    parser.add_argument("--work-name", help="Override work name (defaults to PDF stem)")
    parser.add_argument("--database-url", help="Postgres URL; defaults to DATABASE_URL env var")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text = read_pdf_text(pdf_path)
    work_name = args.work_name or pdf_path.stem
    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not provided; set env or pass --database-url")

    engine, SessionLocal = create_engine_and_session(database_url)
    session = SessionLocal()

    extractor = MockExtractor()  # swap with LLMExtractor for real extraction
    builder = GraphBuilder(session=session, extractor=extractor)

    print(f"Ingesting {pdf_path} as work '{work_name}'...")
    builder.build_from_text(text, work_name=work_name, character=args.character)
    print("Done. Current graph contents:")
    print_nodes_and_edges(session)


if __name__ == "__main__":
    main()
