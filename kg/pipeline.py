
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from sqlalchemy import func, select, or_
from sqlalchemy.orm import Session
from pydantic import ValidationError

from .schema import KGBasicNode, KGEdge, SourceSegment
from .types import EdgeCandidate, NodeCandidate, Segment, NODE_TYPE_VALUES, RELATION_VALUES

logger = logging.getLogger(__name__)


@dataclass
class GraphExtractionResult:
    nodes: List[NodeCandidate]
    edges: List[EdgeCandidate]


DEFAULT_RELATIONS: tuple[str, ...] = RELATION_VALUES


class BaseExtractor(Protocol):
    def extract(
        self,
        segments: List[Segment],
        character: Optional[str] = None,
        existing_context: Optional[str] = None,
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
        allow_open_relations: bool = True,
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
        self.allow_open_relations = allow_open_relations

    def extract(
        self, segments: List[Segment], character: Optional[str] = None, existing_context: Optional[str] = None
    ) -> GraphExtractionResult:
        if len(segments) <= self.batch_threshold:
            return self._extract_single(segments, character, existing_context)
        return self._extract_batched(segments, character, existing_context)

    def _extract_single(
        self, segments: List[Segment], character: Optional[str], existing_context: Optional[str]
    ) -> GraphExtractionResult:
        payload = self._invoke_llm(segments, character, existing_context)
        return self._result_from_payload(payload)

    def _extract_batched(
        self, segments: List[Segment], character: Optional[str], existing_context: Optional[str]
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
                executor.submit(self._invoke_llm, batch, character, existing_context): index
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

    def _invoke_llm(self, segments: List[Segment], character: Optional[str], existing_context: Optional[str]) -> dict:
        prompt = self._build_prompt(segments, character, existing_context)
        response_format = self._build_response_format()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format,
        )
        content = response.choices[0].message.content
        return json.loads(content)

    def _result_from_payload(self, data: dict) -> GraphExtractionResult:
        raw_nodes = data.get("nodes", []) or []
        raw_edges = data.get("edges", []) or []

        nodes: List[NodeCandidate] = [NodeCandidate(**node) for node in self._normalize_nodes(raw_nodes)]
        edges: List[EdgeCandidate] = [EdgeCandidate(**edge) for edge in self._normalize_edges(raw_edges)]

        logger.debug(
            "LLMExtractor normalization: raw nodes=%s -> kept=%s, raw edges=%s -> kept=%s",
            len(raw_nodes),
            len(nodes),
            len(raw_edges),
            len(edges),
        )
        return GraphExtractionResult(nodes=nodes, edges=edges)

    def _build_prompt(
        self, segments: List[Segment], character: Optional[str] = None, existing_context: Optional[str] = None
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
        context_section = ""
        if existing_context:
            context_section = "Existing graph context (do not duplicate; prefer canonical relations):\n" + existing_context + "\n\n"

        return (
            "Extract both graph facts (entities and their relations) and episodic precedents/principles so the character/entity can be represented holistically.\n"
            f"Known character focus: {character or 'none specified'}.\n"
            + context_section +
            "Return strictly valid JSON with top-level keys 'nodes' and 'edges'.\n"
            "nodes: array of {name: string, type: one of [" + node_types + "], summary?: string, alias_names?: string[], meta?: object, source_id?: string}\n"
            "  - Episode nodes: set name to a short episode label; include meta.context, meta.tension, meta.response, meta.rationale, meta.outcome?, meta.confidence?, meta.canon_status?; attach source_id to the best supporting segment.\n"
            "  - Principle nodes: set name/claim to the principle; include meta.scope, meta.support (episode ids/names or source ids), meta.confidence?, meta.exceptions?\n"
            "  - Factual nodes (Character/Person/Organization/Event/etc.): capture summaries and aliases as usual.\n"
            "edges: array of {from_name: string, to_name: string, relation: one of [" + relation_vocab + "], description?: string, meta?: {nature?: string}, source_id?: string, confidence?: number}\n"
            "  - Use only the provided relation vocabulary; if nothing fits, use RELATES_TO and set meta.original_relation to the raw phrase.\n"
            "Do not invent types outside the allowed list. Do not invent relations outside the allowed list. Always include from_name and to_name for edges. Prefer to include source_id on nodes/edges for provenance.\n"
            "Text segments:\n" + "\n".join(text_blocks)
        )

    def _build_response_format(self) -> dict:
        """Structured output schema for OpenAI JSON schema mode."""
        relation_vocab = self.relations or list(DEFAULT_RELATIONS)
        allowed_types = sorted(self.allowed_types)
        generic_types = [t for t in allowed_types if t not in ("Episode", "Principle")]

        episode_node_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"const": "Episode"},
                "summary": {"type": "string"},
                "alias_names": {"type": "array", "items": {"type": "string"}},
                "meta": {
                    "type": "object",
                    "properties": {
                        "context": {"type": "string", "description": "Situation attributes relevant to decision-making."},
                        "tension": {"type": "string", "description": "Core dilemma, tradeoff, or uncertainty."},
                        "response": {"type": "string", "description": "What the agent did/said/decided."},
                        "rationale": {"type": "string", "description": "Why the response occurred (explicit/inferred)."},
                        "outcome": {"type": "string"},
                        "confidence": {"type": "number"},
                        "canon_status": {"type": "string", "description": "canonical | provisional | disputed"},
                    },
                    "required": ["context", "response"],
                    "additionalProperties": True,
                },
                "source_id": {"type": "string"},
            },
            "required": ["name", "type", "meta"],
            "additionalProperties": False,
        }

        principle_node_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Principle claim"},
                "type": {"const": "Principle"},
                "summary": {"type": "string"},
                "alias_names": {"type": "array", "items": {"type": "string"}},
                "meta": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string", "description": "Where/when it applies."},
                        "support": {"type": "array", "items": {"type": "string"}, "description": "Episodes or source ids justifying the principle."},
                        "exceptions": {"type": "array", "items": {"type": "string"}, "description": "Episodes where the principle fails."},
                        "confidence": {"type": "number"},
                    },
                    "required": ["scope"],
                    "additionalProperties": True,
                },
                "source_id": {"type": "string"},
            },
            "required": ["name", "type", "meta"],
            "additionalProperties": False,
        }

        generic_node_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": generic_types},
                "summary": {"type": "string"},
                "alias_names": {"type": "array", "items": {"type": "string"}},
                "meta": {"type": "object", "additionalProperties": True},
                "source_id": {"type": "string"},
            },
            "required": ["name", "type"],
            "additionalProperties": False,
        }

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
                                "oneOf": [
                                    episode_node_schema,
                                    principle_node_schema,
                                    generic_node_schema,
                                ],
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
            if "meta" not in ecopy:
                ecopy["meta"] = {}
            if "relation" in ecopy and isinstance(ecopy["relation"], str):
                sanitized = re.sub(r"[^A-Za-z0-9]+", "_", ecopy["relation"]).strip("_").upper()
                if self.relations and sanitized not in allowed_relations:
                    if self.allow_open_relations:
                        ecopy["meta"].setdefault("original_relation", sanitized)
                        sanitized = "RELATES_TO"
                    else:
                        logger.warning("Skipping edge with unsupported relation: %s (from=%s to=%s raw=%s)", sanitized, ecopy.get("from_name"), ecopy.get("to_name"), edge)
                        continue
                ecopy["relation"] = sanitized
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
        self, segments: List[Segment], character: Optional[str] = None, existing_context: Optional[str] = None
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
        character_key: str = "default",
    ) -> GraphExtractionResult | tuple[GraphExtractionResult, List[Segment]]:
        segments = segment_text(
            text=text,
            work_name=work_name,
            location_prefix=location_prefix,
        )
        work_node = self._ensure_work_node(work_name, work_meta, character_key)
        self._store_segments(segments, work_node.id)

        existing_context = None
        if isinstance(self.extractor, LLMExtractor):
            existing_context = self._build_existing_context(character_key, focus_name=character)

        extraction = self.extractor.extract(segments, character=character, existing_context=existing_context)
        # stamp character_key on candidates if missing
        for n in extraction.nodes:
            if not n.character_key:
                n.character_key = character_key
        for e in extraction.edges:
            if not e.character_key:
                e.character_key = character_key
        self._upsert_nodes(extraction.nodes)
        self._upsert_edges(extraction.edges)
        self.session.commit()
        if return_segments:
            return extraction, segments
        return extraction

    def _ensure_work_node(self, work_name: str, work_meta: Optional[dict], character_key: str) -> KGBasicNode:
        existing = self.session.scalar(
            select(KGBasicNode).where(
                func.lower(KGBasicNode.name) == work_name.lower(),
                KGBasicNode.type == "Work",
                KGBasicNode.character_key == character_key,
            )
        )
        if existing:
            return existing
        node = KGBasicNode(name=work_name, type="Work", meta=work_meta or {}, character_key=character_key)
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
            node = self._get_node_by_name_and_type(candidate.name, candidate.type, candidate.character_key or "default")
            if not node:
                node = KGBasicNode(
                    name=candidate.name,
                    type=candidate.type,
                    character_key=candidate.character_key or "default",
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
                edge.from_name, edge.character_key or "default"
            )
            to_node = name_cache.get(edge.to_name) or self._get_node_by_name(
                edge.to_name, edge.character_key or "default"
            )
            if not from_node or not to_node:
                logger.warning("Skipping edge; missing nodes %s -> %s", edge.from_name, edge.to_name)
                continue
            name_cache[edge.from_name] = from_node
            name_cache[edge.to_name] = to_node
            existing = self.session.scalar(
                select(KGEdge).where(
                    KGEdge.character_key == (edge.character_key or "default"),
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
                    character_key=edge.character_key or "default",
                    from_id=from_node.id,
                    to_id=to_node.id,
                    relation=edge.relation,
                    description=edge.description,
                    weight=edge.confidence,
                    source_ids=[edge.source_id] if edge.source_id else [],
                )
            )
        self.session.flush()

    def _build_existing_context(self, character_key: str, focus_name: Optional[str]) -> Optional[str]:
        """
        Build a compact textual summary of existing graph facts to discourage duplicates.
        """
        if not hasattr(self, "session") or self.session is None:
            return None
        nodes_query = select(KGBasicNode).where(KGBasicNode.character_key == character_key)
        if focus_name:
            nodes_query = nodes_query.where(KGBasicNode.name.ilike(f"%{focus_name}%"))
        nodes = self.session.scalars(nodes_query.limit(30)).all()
        node_lines = [
            f"[{n.type}] {n.name} aliases={n.alias_names or []} summary={n.summary or ''}"
            for n in nodes
        ]

        edges_query = select(KGEdge).where(KGEdge.character_key == character_key)
        if nodes:
            node_ids = [n.id for n in nodes]
            edges_query = edges_query.where(
                or_(KGEdge.from_id.in_(node_ids), KGEdge.to_id.in_(node_ids))
            )
        edges = self.session.scalars(edges_query.limit(50)).all()
        id_to_name = {str(n.id): n.name for n in nodes}
        edge_lines = [
            f"{id_to_name.get(str(e.from_id), str(e.from_id))} -[{e.relation}]-> {id_to_name.get(str(e.to_id), str(e.to_id))}"
            for e in edges
        ]
        if not node_lines and not edge_lines:
            return None
        return "\n".join(node_lines + edge_lines)

    def _get_node_by_name_and_type(
        self, name: str, node_type: str, character_key: str
    ) -> Optional[KGBasicNode]:
        return self.session.scalar(
            select(KGBasicNode).where(
                func.lower(KGBasicNode.name) == name.lower(),
                KGBasicNode.type == node_type,
                KGBasicNode.character_key == character_key,
            )
        )

    def _get_node_by_name(self, name: str, character_key: str) -> Optional[KGBasicNode]:
        return self.session.scalar(
            select(KGBasicNode).where(
                func.lower(KGBasicNode.name) == name.lower(),
                KGBasicNode.character_key == character_key,
            )
        )
