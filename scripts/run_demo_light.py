"""
Lightweight demo: no database required.

Reads a text or PDF file, segments it, runs the extractor (MockExtractor by default),
and prints segments + extracted nodes/edges.

Usage:
  python scripts/run_demo_light.py --text test_files/demo_story.txt --character "Aurelia Maren"
  python scripts/run_demo_light.py --pdf test_files/demo_story.pdf --character "Aurelia Maren"
"""

import argparse
from pathlib import Path

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - pypdf is optional
    PdfReader = None

from kg.pipeline import MockExtractor, segment_text


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_pdf(path: Path) -> str:
    if PdfReader is None:
        raise ImportError("pypdf is not installed. Install it or pass --text instead of --pdf.")
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def main():
    parser = argparse.ArgumentParser(description="Lightweight KG demo (no DB).")
    parser.add_argument("--text", help="Path to a text file to ingest.")
    parser.add_argument("--pdf", help="Path to a PDF file to ingest.")
    parser.add_argument("--character", help="Optional focus character name.", default=None)
    parser.add_argument("--work-name", help="Optional work name; defaults to filename stem.")
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
    segments = segment_text(content, work_name=work_name)
    extractor = MockExtractor()
    extraction = extractor.extract(segments, character=args.character)

    print(f"Work: {work_name}")
    print(f"Segments: {len(segments)}")
    for seg in segments:
        print(f"[{seg.id}] ({seg.location}) {seg.content[:120]}{'...' if len(seg.content) > 120 else ''}")

    print("\nNodes:")
    for node in extraction.nodes:
        print(f"- {node.type}: {node.name} (aliases={node.alias_names}, src={node.source_id})")

    print("\nEdges:")
    for edge in extraction.edges:
        print(
            f"- {edge.from_name} -[{edge.relation}]-> {edge.to_name} "
            f"(src={edge.source_id}, conf={edge.confidence})"
        )


if __name__ == "__main__":
    main()
