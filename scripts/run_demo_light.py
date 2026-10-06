"""
Lightweight demo: no database required.

Reads a text or PDF file, segments it, runs the extractor, and prints segments + extracted nodes/edges.
By default uses MockExtractor. Use --use-llm to call OpenAI (requires OPENAI_API_KEY).

Usage (from the repository root):
  PYTHONPATH=. python scripts/run_demo_light.py --text test_files/demo_story.txt --character "Aurelia Maren"
  PYTHONPATH=. python scripts/run_demo_light.py --pdf test_files/demo_story.pdf --character "Aurelia Maren"
  PYTHONPATH=. python scripts/run_demo_light.py --text test_files/demo_story.txt --use-llm --character "Aurelia Maren"
"""

import argparse
import os
from pathlib import Path
from typing import Optional

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - pypdf is optional
    PdfReader = None

from kg.pipeline import DEFAULT_RELATIONS, LLMExtractor, MockExtractor, segment_text


def format_source(segment_lookup, source_id: Optional[str]) -> str:
    if not source_id:
        return "(none)"
    segment = segment_lookup.get(source_id)
    if not segment:
        return source_id
    snippet = segment.content.strip()
    if len(snippet) > 160:
        snippet = snippet[:160].rstrip() + "..."
    return f"{source_id}: {snippet}"


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
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use OpenAI via LLMExtractor instead of MockExtractor (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DEMO_LLM_MODEL", "gpt-4o-mini"),
        help="LLM model name (default: gpt-4o-mini or $DEMO_LLM_MODEL).",
    )
    parser.add_argument(
        "--relations",
        nargs="*",
        default=list(DEFAULT_RELATIONS),
        help="Relation vocabulary for the extractor.",
    )
    parser.add_argument(
        "--llm-batch-threshold",
        type=int,
        default=32,
        help="Max segments per single LLM prompt before batching kicks in.",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=10,
        help="Number of segments per LLM batch when pagination is enabled.",
    )
    parser.add_argument(
        "--llm-batch-overlap",
        type=int,
        default=1,
        help="How many segments to overlap between consecutive LLM batches.",
    )
    parser.add_argument(
        "--llm-max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel LLM requests when batching.",
    )
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
    segment_lookup = {segment.id: segment for segment in segments}
    if args.use_llm:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when using --use-llm")
        client = OpenAI(api_key=api_key)
        extractor = LLMExtractor(
            client=client,
            model=args.model,
            relations=args.relations,
            batch_threshold=args.llm_batch_threshold,
            batch_size=args.llm_batch_size,
            batch_overlap=args.llm_batch_overlap,
            max_workers=args.llm_max_workers,
        )
        extractor_name = f"LLMExtractor(model={args.model})"
    else:
        extractor = MockExtractor()
        extractor_name = "MockExtractor"
    extraction = extractor.extract(segments, character=args.character)

    print(f"Work: {work_name}")
    print(f"Extractor: {extractor_name}")
    print(f"Segments: {len(segments)}")
    for seg in segments:
        print(f"[{seg.id}] ({seg.location}) {seg.content[:120]}{'...' if len(seg.content) > 120 else ''}")

    print("\nNodes:")
    for node in extraction.nodes:
        source_display = format_source(segment_lookup, node.source_id)
        print(f"- {node.type}: {node.name} (aliases={node.alias_names})")
        print(f"    source: {source_display}")

    print("\nEdges:")
    for edge in extraction.edges:
        source_display = format_source(segment_lookup, edge.source_id)
        print(
            f"- {edge.from_name} -[{edge.relation}]-> {edge.to_name} "
            f"(conf={edge.confidence})"
        )
        print(f"    source: {source_display}")


if __name__ == "__main__":
    main()
