
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .schema import KGBasicNode, KGEdge, SourceSegment
from .types import EdgeCandidate, NodeCandidate, Segment

logger = logging.getLogger(__name__)


@dataclass
class GraphExtractionResult:
    nodes: List[NodeCandidate]
    edges: List[EdgeCandidate]


class BaseExtractor(Protocol):
    def extract(
        self, segments: List[Segment], character: Optional[str] = None
    ) -> GraphExtractionResult:
        ...


def segment_text(
    text: str,
    work_name: str,
    location_prefix: Optional[str] = None,
    max_words: int = 120,
    overlap_sentences: int = 1,
) -> List[Segment]:
    """
    Sentence-aware segmenter that packs whole sentences into chunks up to `max_words`,
    never splitting a sentence. Optional `overlap_sentences` (default 1) carries context between chunks.
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    segments: List[Segment] = []
    for para_index, paragraph in enumerate(raw_paragraphs):
        sentences = split_sentences(paragraph)
        chunk: List[str] = []
        chunk_words = 0
        chunk_index = 0
        for sentence in sentences:
            sent_words = len(sentence.split())
            # start a new chunk if adding this sentence would exceed the limit
            if chunk and chunk_words + sent_words > max_words:
                segment_id = f"{slugify(work_name)}-{para_index}-{chunk_index}"
                segments.append(
                    Segment(
                        id=segment_id,
                        work_name=work_name,
                        location=f"{location_prefix or 'p'}{para_index}-{chunk_index}",
                        content=" ".join(chunk),
                    )
                )
                chunk_index += 1
                if overlap_sentences > 0:
                    chunk = chunk[-overlap_sentences:]
                else:
                    chunk = []
                chunk_words = sum(len(s.split()) for s in chunk)
            chunk.append(sentence)
            chunk_words += sent_words
        if chunk:
            segment_id = f"{slugify(work_name)}-{para_index}-{chunk_index}"
            segments.append(
                Segment(
                    id=segment_id,
                    work_name=work_name,
                    location=f"{location_prefix or 'p'}{para_index}-{chunk_index}",
                    content=" ".join(chunk),
                )
            )
    return segments


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def split_sentences(paragraph: str) -> List[str]:
    """
    Lightweight sentence splitter using punctuation boundaries.
    Keeps punctuation attached and trims whitespace.
    """
    raw = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [s.strip() for s in raw if s.strip()]


class LLMExtractor(BaseExtractor):
    """
    Minimal LLM wrapper that expects a JSON payload with nodes and edges.

    The model must return:
    {
      "nodes": [{"name": "...", "type": "...", "summary": "...", "alias_names": [], "meta": {}, "source_id": "..."}],
      "edges": [{"from_name": "...", "to_name": "...", "relation": "...", "description": "...", "source_id": "...", "confidence": 0.8}]
    }
    """

    def __init__(self, client, model: str, relations: Optional[List[str]] = None):
        self.client = client
        self.model = model
        self.relations = relations or []

    def extract(
        self, segments: List[Segment], character: Optional[str] = None
    ) -> GraphExtractionResult:
        prompt = self._build_prompt(segments, character)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        nodes = [NodeCandidate(**node) for node in data.get("nodes", [])]
        edges = [EdgeCandidate(**edge) for edge in data.get("edges", [])]
        return GraphExtractionResult(nodes=nodes, edges=edges)

    def _build_prompt(
        self, segments: List[Segment], character: Optional[str] = None
    ) -> str:
        text_blocks = []
        for segment in segments:
            text_blocks.append(
                f"[{segment.id}] ({segment.location}) {segment.content}"
            )
        relation_vocab = ", ".join(self.relations) if self.relations else "see schema"
        return (
            "Extract graph facts about the character and related entities.\n"
            f"Known character focus: {character or 'none specified'}.\n"
            "Return JSON with keys nodes and edges.\n"
            f"Relations must be one of: {relation_vocab}.\n"
            "Text segments:\n" + "\n".join(text_blocks)
        )


class MockExtractor(BaseExtractor):
    """Simple extractor useful for tests; creates a Work node and SourceSegment edges."""

    def extract(
        self, segments: List[Segment], character: Optional[str] = None
    ) -> GraphExtractionResult:
        nodes: List[NodeCandidate] = []
        edges: List[EdgeCandidate] = []
        if character:
            nodes.append(
                NodeCandidate(
                    name=character,
                    type="Character",
                    summary=f"Auto-created focus character {character}",
                )
            )
        for segment in segments:
            nodes.append(
                NodeCandidate(
                    name=segment.work_name,
                    type="Work",
                    summary="Document source",
                    source_id=segment.id,
                )
            )
            if character:
                edges.append(
                    EdgeCandidate(
                        from_name=character,
                        to_name=segment.work_name,
                        relation="REFERENCES",
                        description="Character discussed in this work",
                        source_id=segment.id,
                        confidence=0.3,
                    )
                )
        return GraphExtractionResult(nodes=nodes, edges=edges)


class GraphBuilder:
    def __init__(self, session: Session, extractor: BaseExtractor):
        self.session = session
        self.extractor = extractor

    def build_from_text(
        self,
        text: str,
        work_name: str,
        work_meta: Optional[dict] = None,
        character: Optional[str] = None,
        location_prefix: Optional[str] = None,
        return_segments: bool = False,
    ) -> GraphExtractionResult | tuple[GraphExtractionResult, List[Segment]]:
        segments = segment_text(
            text=text,
            work_name=work_name,
            location_prefix=location_prefix,
        )
        work_node = self._ensure_work_node(work_name, work_meta)
        self._store_segments(segments, work_node.id)

        extraction = self.extractor.extract(segments, character=character)
        self._upsert_nodes(extraction.nodes)
        self._upsert_edges(extraction.edges)
        self.session.commit()
        if return_segments:
            return extraction, segments
        return extraction

    def _ensure_work_node(self, work_name: str, work_meta: Optional[dict]) -> KGBasicNode:
        existing = self.session.scalar(
            select(KGBasicNode).where(
                func.lower(KGBasicNode.name) == work_name.lower(),
                KGBasicNode.type == "Work",
            )
        )
        if existing:
            return existing
        node = KGBasicNode(name=work_name, type="Work", meta=work_meta or {})
        self.session.add(node)
        self.session.flush()
        return node

    def _store_segments(self, segments: Iterable[Segment], work_id) -> None:
        for segment in segments:
            existing = self.session.get(SourceSegment, segment.id)
            if existing:
                continue
            self.session.add(
                SourceSegment(
                    id=segment.id,
                    work_id=work_id,
                    location=segment.location,
                    content=segment.content,
                    meta=segment.meta,
                )
            )

    def _upsert_nodes(self, nodes: Iterable[NodeCandidate]) -> None:
        for candidate in nodes:
            node = self._get_node_by_name_and_type(candidate.name, candidate.type)
            if not node:
                node = KGBasicNode(
                    name=candidate.name,
                    type=candidate.type,
                    alias_names=candidate.alias_names,
                    summary=candidate.summary,
                    meta=candidate.meta,
                )
                self.session.add(node)
                self.session.flush()
            else:
                node.alias_names = list(
                    set((node.alias_names or []) + list(candidate.alias_names))
                )
                if candidate.summary and not node.summary:
                    node.summary = candidate.summary
                node.meta = {**(node.meta or {}), **(candidate.meta or {})}
            if candidate.source_id:
                node.source_ids = list(set((node.source_ids or []) + [candidate.source_id]))
        self.session.flush()

    def _upsert_edges(self, edges: Iterable[EdgeCandidate]) -> None:
        name_cache = {}
        for edge in edges:
            from_node = name_cache.get(edge.from_name) or self._get_node_by_name(
                edge.from_name
            )
            to_node = name_cache.get(edge.to_name) or self._get_node_by_name(
                edge.to_name
            )
            if not from_node or not to_node:
                logger.warning("Skipping edge; missing nodes %s -> %s", edge.from_name, edge.to_name)
                continue
            name_cache[edge.from_name] = from_node
            name_cache[edge.to_name] = to_node
            existing = self.session.scalar(
                select(KGEdge).where(
                    KGEdge.from_id == from_node.id,
                    KGEdge.to_id == to_node.id,
                    KGEdge.relation == edge.relation,
                )
            )
            if existing:
                if edge.description:
                    existing.description = edge.description
                if edge.confidence:
                    existing.weight = edge.confidence
                if edge.source_id:
                    existing.source_ids = list(
                        set((existing.source_ids or []) + [edge.source_id])
                    )
                continue
            self.session.add(
                KGEdge(
                    from_id=from_node.id,
                    to_id=to_node.id,
                    relation=edge.relation,
                    description=edge.description,
                    weight=edge.confidence,
                    source_ids=[edge.source_id] if edge.source_id else [],
                )
            )
        self.session.flush()

    def _get_node_by_name_and_type(
        self, name: str, node_type: str
    ) -> Optional[KGBasicNode]:
        return self.session.scalar(
            select(KGBasicNode).where(
                func.lower(KGBasicNode.name) == name.lower(),
                KGBasicNode.type == node_type,
            )
        )

    def _get_node_by_name(self, name: str) -> Optional[KGBasicNode]:
        return self.session.scalar(
            select(KGBasicNode).where(func.lower(KGBasicNode.name) == name.lower())
        )
