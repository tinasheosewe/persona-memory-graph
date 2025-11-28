
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .schema import KGBasicNode, KGEdge, SourceSegment
from .types import EdgeCandidate, NodeCandidate, Segment, NODE_TYPE_VALUES

logger = logging.getLogger(__name__)


@dataclass
class GraphExtractionResult:
    nodes: List[NodeCandidate]
    edges: List[EdgeCandidate]


DEFAULT_RELATIONS: tuple[str, ...] = (
    "RELATES_TO",
    "BELIEVES_IN",
    "OPPOSES",
    "INFLUENCED_BY",
    "FRIEND_OF",
    "ENEMY_OF",
    "MENTORED_BY",
    "MEMBER_OF",
    "PARTICIPATED_IN",
    "OCCURRED_AT",
    "OCCURRED_DURING",
    "WROTE",
    "DISCUSSES",
    "REFERENCES",
    "CONTRADICTS",
    "SUPPORTS",
    "INSPIRED_BY",
    "FOUNDED",
)


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

    def __init__(
        self,
        client,
        model: str,
        relations: Optional[List[str]] = None,
        batch_threshold: int = 32,
        batch_size: int = 10,
        batch_overlap: int = 1,
        max_workers: int = 4,
    ):
        self.client = client
        self.model = model
        self.relations = list(relations or DEFAULT_RELATIONS)
        self.allowed_types = set(NODE_TYPE_VALUES)
        self.batch_threshold = max(1, batch_threshold)
        self.batch_size = max(1, batch_size)
        if batch_overlap >= self.batch_size:
            logger.warning(
                "batch_overlap (%s) >= batch_size (%s); reducing overlap to batch_size-1",
                batch_overlap,
                self.batch_size,
            )
        self.batch_overlap = max(0, min(batch_overlap, self.batch_size - 1))
        self.max_workers = max(1, max_workers)

    def extract(
        self, segments: List[Segment], character: Optional[str] = None
    ) -> GraphExtractionResult:
        if len(segments) <= self.batch_threshold:
            return self._extract_single(segments, character)
        return self._extract_batched(segments, character)

    def _extract_single(
        self, segments: List[Segment], character: Optional[str]
    ) -> GraphExtractionResult:
        payload = self._invoke_llm(segments, character)
        return self._result_from_payload(payload)

    def _extract_batched(
        self, segments: List[Segment], character: Optional[str]
    ) -> GraphExtractionResult:
        batches = self._create_batches(segments)
        logger.info(
            "LLMExtractor batching %s segments into %s requests (size=%s, overlap=%s)",
            len(segments),
            len(batches),
            self.batch_size,
            self.batch_overlap,
        )
        batch_payloads: List[Optional[dict]] = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._invoke_llm, batch, character): index
                for index, batch in enumerate(batches)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                batch_payloads[index] = future.result()
        batch_results = [
            self._result_from_payload(payload)
            for payload in batch_payloads
            if payload is not None
        ]
        return self._merge_results(batch_results)

    def _invoke_llm(self, segments: List[Segment], character: Optional[str]) -> dict:
        prompt = self._build_prompt(segments, character)
        response_format = self._build_response_format()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format,
        )
        content = response.choices[0].message.content
        return json.loads(content)

    def _result_from_payload(self, data: dict) -> GraphExtractionResult:
        nodes = [NodeCandidate(**node) for node in self._normalize_nodes(data.get("nodes", []))]
        edges = [EdgeCandidate(**edge) for edge in self._normalize_edges(data.get("edges", []))]
        return GraphExtractionResult(nodes=nodes, edges=edges)

    def _build_prompt(
        self, segments: List[Segment], character: Optional[str] = None
    ) -> str:
        text_blocks = []
        for segment in segments:
            text_blocks.append(
                f"[{segment.id}] ({segment.location}) {segment.content}"
            )
        relation_vocab = (
            ", ".join(self.relations)
            if self.relations
            else ", ".join(DEFAULT_RELATIONS)
        )
        node_types = ", ".join(sorted(self.allowed_types))
        return (
            "Extract graph facts about the character and related entities.\n"
            f"Known character focus: {character or 'none specified'}.\n"
            "Return strictly valid JSON with top-level keys 'nodes' and 'edges'.\n"
            "nodes: array of {name: string, type: one of [" + node_types + "], summary?: string, alias_names?: string[], meta?: object, source_id?: string}\n"
            "edges: array of {from_name: string, to_name: string, relation: one of [" + relation_vocab + "], description?: string, meta?: {nature?: string}, source_id?: string, confidence?: number}\n"
            "Do not invent types outside the allowed list. Always include from_name and to_name for edges. Use the provided relation vocabulary only.\n"
            "Text segments:\n" + "\n".join(text_blocks)
        )

    def _build_response_format(self) -> dict:
        """Structured output schema for OpenAI JSON schema mode."""
        relation_vocab = self.relations or list(DEFAULT_RELATIONS)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "graph_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string", "enum": sorted(self.allowed_types)},
                                    "summary": {"type": "string"},
                                    "alias_names": {"type": "array", "items": {"type": "string"}},
                                    "meta": {"type": "object", "additionalProperties": True},
                                    "source_id": {"type": "string"},
                                },
                                "required": ["name", "type"],
                                "additionalProperties": False,
                            },
                            "default": [],
                        },
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from_name": {"type": "string"},
                                    "to_name": {"type": "string"},
                                    "relation": {"type": "string", "enum": relation_vocab},
                                    "description": {"type": "string"},
                                    "source_id": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "meta": {
                                        "type": "object",
                                        "additionalProperties": True,
                                        "properties": {
                                            "nature": {"type": "string"},
                                        },
                                    },
                                },
                                "required": ["from_name", "to_name", "relation"],
                                "additionalProperties": False,
                            },
                            "default": [],
                        },
                    },
                    "required": ["nodes", "edges"],
                    "additionalProperties": False,
                },
            },
        }

    def _normalize_nodes(self, raw_nodes: list) -> List[dict]:
        normalized = []
        for node in raw_nodes:
            node_copy = dict(node)
            if "name" not in node_copy and "id" in node_copy:
                node_copy["name"] = node_copy["id"]
            if "type" in node_copy and isinstance(node_copy["type"], str):
                tval = node_copy["type"].strip()
                tcap = tval[:1].upper() + tval[1:]
                if tcap not in self.allowed_types:
                    raise ValueError(f"LLM returned unsupported node type '{tval}'")
                node_copy["type"] = tcap
            normalized.append(node_copy)
        return normalized

    def _normalize_edges(self, raw_edges: list) -> List[dict]:
        normalized_edges = []
        allowed_relations = set(self.relations)
        for edge in raw_edges:
            ecopy = dict(edge)
            if "from_name" not in ecopy:
                for key in ("source", "from", "actor"):
                    if key in ecopy:
                        ecopy["from_name"] = ecopy[key]
                        break
            if "to_name" not in ecopy:
                for key in ("target", "to", "object"):
                    if key in ecopy:
                        ecopy["to_name"] = ecopy[key]
                        break
            if "relation" in ecopy and isinstance(ecopy["relation"], str):
                sanitized = re.sub(r"[^A-Za-z0-9]+", "_", ecopy["relation"]).strip("_").upper()
                ecopy["relation"] = sanitized
            if self.relations and ecopy.get("relation") not in allowed_relations:
                logger.warning("Skipping edge with unsupported relation: %s", ecopy.get("relation"))
                continue
            if "confidence" not in ecopy and "weight" in ecopy:
                ecopy["confidence"] = ecopy["weight"]
            if "meta" not in ecopy:
                ecopy["meta"] = {}
            if "from_name" not in ecopy or "to_name" not in ecopy:
                raise ValueError(f"LLM returned edge missing endpoints: {ecopy}")
            normalized_edges.append(ecopy)
        return normalized_edges

    def _create_batches(self, segments: Sequence[Segment]) -> List[List[Segment]]:
        if len(segments) <= self.batch_size:
            return [list(segments)]
        stride = max(1, self.batch_size - self.batch_overlap)
        seg_list = list(segments)
        batches: List[List[Segment]] = []
        start = 0
        while start < len(seg_list):
            end = min(len(seg_list), start + self.batch_size)
            batches.append(seg_list[start:end])
            if end >= len(seg_list):
                break
            start += stride
        return batches

    def _merge_results(
        self, results: Sequence[GraphExtractionResult]
    ) -> GraphExtractionResult:
        node_map: Dict[Tuple[str, str], NodeCandidate] = {}
        edge_map: Dict[Tuple[str, str, str], EdgeCandidate] = {}
        for result in results:
            self._merge_nodes(result.nodes, node_map)
            self._merge_edges(result.edges, edge_map)
        merged_nodes = sorted(
            node_map.values(),
            key=lambda n: (n.type, n.name.lower()),
        )
        merged_edges = sorted(
            edge_map.values(),
            key=lambda e: (e.from_name.lower(), e.to_name.lower(), e.relation),
        )
        return GraphExtractionResult(nodes=merged_nodes, edges=merged_edges)

    def _merge_nodes(
        self,
        nodes: Iterable[NodeCandidate],
        node_map: Dict[Tuple[str, str], NodeCandidate],
    ) -> None:
        for node in nodes:
            key = (node.name.lower(), node.type)
            if key not in node_map:
                node_map[key] = node
                continue
            existing = node_map[key]
            alias_names = sorted(
                set((existing.alias_names or [])) | set(node.alias_names or [])
            )
            summary = existing.summary or node.summary
            meta = {**(existing.meta or {}), **(node.meta or {})}
            source_id = existing.source_id or node.source_id
            node_map[key] = NodeCandidate(
                name=existing.name,
                type=existing.type,
                summary=summary,
                alias_names=alias_names,
                meta=meta,
                source_id=source_id,
            )

    def _merge_edges(
        self,
        edges: Iterable[EdgeCandidate],
        edge_map: Dict[Tuple[str, str, str], EdgeCandidate],
    ) -> None:
        for edge in edges:
            key = (edge.from_name.lower(), edge.to_name.lower(), edge.relation)
            if key not in edge_map:
                edge_map[key] = edge
                continue
            existing = edge_map[key]
            description = existing.description or edge.description
            meta = {**(existing.meta or {}), **(edge.meta or {})}
            source_id = existing.source_id or edge.source_id
            confidences = [c for c in (existing.confidence, edge.confidence) if c is not None]
            confidence = max(confidences) if confidences else None
            edge_map[key] = EdgeCandidate(
                from_name=existing.from_name,
                to_name=existing.to_name,
                relation=existing.relation,
                description=description,
                source_id=source_id,
                confidence=confidence,
                meta=meta,
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
