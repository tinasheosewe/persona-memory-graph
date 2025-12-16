"""
Ingest a text or PDF into Neo4j (skipping Postgres) and print nodes/edges.

Requires:
- NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
- Optional: OPENAI_API_KEY to use LLMExtractor; otherwise uses MockExtractor.

Usage:
  python scripts/ingest_to_neo4j.py --text test_files/demo_story.txt --character Aurelia
  python scripts/ingest_to_neo4j.py --pdf test_files/demo_story.pdf --character Aurelia
"""

import argparse
import logging
import os
from pathlib import Path

from neo4j import GraphDatabase

from kg.neo4j_memgraph import CypherGraphBuilder
from kg.pipeline import LLMExtractor, MockExtractor, segment_text

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_pdf(path: Path) -> str:
    if PdfReader is None:
        raise ImportError("pypdf is not installed. Install it or pass --text instead of --pdf.")
    reader = PdfReader(str(path))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n\n".join(pages)


def choose_extractor():
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if api_key:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        return LLMExtractor(client=client, model=model)
    return MockExtractor()


def main():
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOGLEVEL", "INFO").upper(), logging.INFO)
    )
    parser = argparse.ArgumentParser(description="Ingest into Neo4j and print nodes/edges.")
    parser.add_argument("--text", help="Path to text file")
    parser.add_argument("--pdf", help="Path to PDF file")
    parser.add_argument("--character", help="Focus character name", default=None)
    parser.add_argument("--work-name", help="Override work name", default=None)
    parser.add_argument("--character-key", help="Partition key", default="default")
    args = parser.parse_args()

    if not args.text and not args.pdf:
        raise ValueError("Provide --text or --pdf")
    source_path = Path(args.text or args.pdf).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if args.pdf:
        content = read_pdf(source_path)
    else:
        content = read_text(source_path)

    work_name = args.work_name or source_path.stem
    extractor = choose_extractor()

    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]
    driver = GraphDatabase.driver(uri, auth=(user, password))
    builder = CypherGraphBuilder(driver=driver, extractor=extractor)

    print(f"Ingesting '{work_name}' into Neo4j with character_key='{args.character_key}'...")
    extraction = builder.build_from_text(
        content,
        work_name=work_name,
        character=args.character,
        character_key=args.character_key,
    )

    print("\nNodes:")
    for node in extraction.nodes:
        print(f"- {node.type} {node.name} (ck={node.character_key} src={node.source_id})")

    print("\nEdges:")
    for edge in extraction.edges:
        print(
            f"- {edge.from_name} -[{edge.relation}]-> {edge.to_name} "
            f"(ck={edge.character_key} src={edge.source_id} meta={edge.meta})"
        )


if __name__ == "__main__":
    main()
